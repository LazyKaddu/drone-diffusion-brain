# models/policy.py

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.encoder import ObservationEncoder
from models.unet1d import ConditionalUnet1D


class DroneDiffusionPolicy(nn.Module):
    """
    Unified Diffusion Policy wrapper for autonomous multirotor flight.
    
    Couples:
      1. Multi-modal state observation encoder (Kinematics / Vision -> Latent c)
      2. 1D Temporal ResNet U-Net with FiLM conditioning (predicts noise eps_theta)
      3. Hugging Face Diffusers Schedulers (DDPM for training, DDIM for low-latency inference)
    
    Shape Conventions:
      - Outside World / Dataset: Actions are [Batch, Time (Ta), Channels (Da)]
      - Internal 1D Convolutions: Actions are [Batch, Channels (Da), Time (Ta)]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.policy.action_dim
        self.pred_horizon = cfg.policy.pred_horizon
        self.obs_horizon = cfg.policy.obs_horizon
        self.exec_horizon = cfg.policy.exec_horizon

        # -------------------------------------------------------------
        # 1. State Observation Encoder
        # -------------------------------------------------------------
        self.encoder = ObservationEncoder(cfg)

        # -------------------------------------------------------------
        # 2. Conditional 1D Temporal U-Net Backbone
        # -------------------------------------------------------------
        self.model = ConditionalUnet1D(
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            cond_dim=cfg.policy.unet.cond_dim,
            diffusion_step_embed_dim=cfg.policy.unet.diffusion_step_embed_dim,
            down_dims=list(cfg.policy.unet.down_dims),
            kernel_size=cfg.policy.unet.kernel_size,
            n_groups=cfg.policy.unet.n_groups,
            dropout=cfg.policy.unet.dropout,
        )

        # -------------------------------------------------------------
        # 3. Diffusion Noise Schedulers
        # -------------------------------------------------------------
        diff_cfg = cfg.policy.diffusion
        
        # Training Scheduler: 100-step DDPM
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff_cfg.train_scheduler.num_train_timesteps,
            beta_start=diff_cfg.train_scheduler.beta_start,
            beta_end=diff_cfg.train_scheduler.beta_end,
            beta_schedule=diff_cfg.train_scheduler.beta_schedule,
            clip_sample=diff_cfg.train_scheduler.clip_sample,
            prediction_type=diff_cfg.train_scheduler.prediction_type,
        )

        # Inference Scheduler: Fast DDIM for sub-50ms execution
        self.eval_scheduler = DDIMScheduler(
            num_train_timesteps=diff_cfg.train_scheduler.num_train_timesteps,
            beta_start=diff_cfg.train_scheduler.beta_start,
            beta_end=diff_cfg.train_scheduler.beta_end,
            beta_schedule=diff_cfg.train_scheduler.beta_schedule,
            clip_sample=diff_cfg.train_scheduler.clip_sample,
            prediction_type=diff_cfg.train_scheduler.prediction_type,
            set_alpha_to_one=False,
            steps_offset=0,
        )
        self.num_inference_timesteps = diff_cfg.inference_scheduler.num_inference_timesteps
        self.eval_scheduler.set_timesteps(self.num_inference_timesteps)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the denoising diffusion MSE loss on a training batch.

        Args:
            batch: Dict containing:
                - 'obs': [Batch, To, obs_dim] or Dict of tensors
                - 'action': [Batch, Ta, action_dim] ground-truth continuous actions

        Returns:
            Scalar MSE loss.
        """
        actions = batch["action"]  # [B, Ta, Da]
        obs = batch["obs"]        # [B, To, Do]
        device = actions.device
        batch_size = actions.shape[0]

        # 1. Encode conditioning state
        cond = self.encoder(obs)  # [B, cond_dim]

        # 2. Sample random Gaussian noise with matching action shape
        noise = torch.randn_like(actions)

        # 3. Sample random diffusion timesteps uniform over [0, T_train)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        # 4. Forward diffusion: Add noise according to schedule alpha_t
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # 5. Temporal dimension transposition: [B, Ta, Da] -> [B, Da, Ta]
        noisy_actions_1d = noisy_actions.transpose(1, 2)

        # 6. Predict noise residual using U-Net
        pred_noise_1d = self.model(
            sample=noisy_actions_1d,
            timestep=timesteps,
            cond=cond,
        )

        # 7. Transpose back: [B, Da, Ta] -> [B, Ta, Da]
        pred_noise = pred_noise_1d.transpose(1, 2)

        # 8. Compute target objective
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else actions
        loss = F.mse_loss(pred_noise, target)

        return loss

    @torch.no_grad()
    def predict_action(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Denoises an action chunk from pure Gaussian noise using accelerated DDIM sampling.

        Args:
            obs: Observation tensor [B, To, obs_dim] or [To, obs_dim] or dict
            num_inference_steps: Number of DDIM steps (defaults to config: 10)

        Returns:
            Clean action chunk [B, Ta, action_dim] in normalized range [-1.0, 1.0]
        """
        # Ensure batch dimension
        if isinstance(obs, torch.Tensor):
            if obs.ndim == 2:
                obs = obs.unsqueeze(0)  # [1, To, obs_dim]
            device = obs.device
            batch_size = obs.shape[0]
        elif isinstance(obs, dict):
            first_val = next(iter(obs.values()))
            if first_val.ndim == (3 if "rgb" not in obs else 4):
                obs = {k: v.unsqueeze(0) for k, v in obs.items()}
            device = first_val.device
            batch_size = first_val.shape[0]
        else:
            raise TypeError(f"Unsupported observation format: {type(obs)}")

        # 1. Encode physical state
        cond = self.encoder(obs)  # [B, cond_dim]

        # 2. Configure DDIM inference schedule
        steps = num_inference_steps or self.num_inference_timesteps
        self.eval_scheduler.set_timesteps(steps)

        # 3. Initialize trajectory from pure Gaussian noise: X_T ~ N(0, I)
        trajectory = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=device,
            dtype=torch.float32,
        )

        # 4. Iterative DDIM Denoising Loop
        for t in self.eval_scheduler.timesteps:
            # Transpose to 1D Conv layout: [B, Ta, Da] -> [B, Da, Ta]
            trajectory_1d = trajectory.transpose(1, 2)

            # Predict noise residual
            model_output_1d = self.model(
                sample=trajectory_1d,
                timestep=t,
                cond=cond,
            )

            # Transpose back: [B, Da, Ta] -> [B, Ta, Da]
            model_output = model_output_1d.transpose(1, 2)

            # Step backward along the DDIM trajectory
            trajectory = self.eval_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=trajectory,
            ).prev_sample

        # 5. Bound predicted actions to normalized limits [-1.0, 1.0]
        trajectory = torch.clamp(trajectory, -1.0, 1.0)
        return trajectory

    @torch.no_grad()
    def get_receding_actions(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        exec_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convenience execution helper for Receding Horizon Control.
        Predicts Ta=16 action sequence and returns only the initial Te steps.

        Returns:
            Action slice [B, Te, action_dim]
        """
        horizon = exec_horizon or self.exec_horizon
        full_chunk = self.predict_action(obs)
        return full_chunk[:, :horizon, :]


if __name__ == "__main__":
    from types import SimpleNamespace

    # Mock Hydra config matching default.yaml
    mock_cfg = SimpleNamespace(
        policy=SimpleNamespace(
            action_dim=4,
            pred_horizon=16,
            obs_horizon=1,
            exec_horizon=4,
            obs_dim=20,
            encoder=SimpleNamespace(
                type="mlp",
                latent_dim=256,
                hidden_dims=[128, 256],
            ),
            unet=SimpleNamespace(
                down_dims=[128, 256, 512],
                kernel_size=5,
                n_groups=8,
                cond_dim=256,
                diffusion_step_embed_dim=128,
                dropout=0.0,
            ),
            diffusion=SimpleNamespace(
                train_scheduler=SimpleNamespace(
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="squaredcos_cap_v2",
                    clip_sample=True,
                    prediction_type="epsilon",
                ),
                inference_scheduler=SimpleNamespace(
                    num_inference_timesteps=10,
                ),
            ),
        )
    )

    policy = DroneDiffusionPolicy(mock_cfg)

    # 1. Verify Training Loss Pass
    dummy_batch = {
        "obs": torch.randn(8, 1, 20),
        "action": torch.randn(8, 16, 4),
    }
    loss = policy.compute_loss(dummy_batch)
    print(f"Training Loss Output: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss computed NaN"

    # 2. Verify Inference Denoising Pass (10 DDIM steps)
    single_obs = torch.randn(1, 20)
    action_chunk = policy.predict_action(single_obs)
    print(f"Inferred Action Chunk Shape: {action_chunk.shape}")
    assert action_chunk.shape == (1, 16, 4), "Output shape mismatch"

    # 3. Verify Receding Horizon Slicing
    exec_actions = policy.get_receding_actions(single_obs)
    print(f"Receding Horizon Slice (Te=4): {exec_actions.shape}")
    assert exec_actions.shape == (1, 4, 4), "Execution slice mismatch"

    print("Policy verification tests passed successfully.")