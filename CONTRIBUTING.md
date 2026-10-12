# Contributing to Drone Diffusion Brain

Thank you for your interest in contributing to **Drone Diffusion Brain**! We welcome contributions from roboticists, machine learning researchers, and software engineers to push the state of the art in generative diffusion policies for autonomous aerial navigation.

---

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Code Quality & Testing](#code-quality--testing)
- [Commit & Pull Request Guidelines](#commit--pull-request-guidelines)
- [License](#license)

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and open to feedback in all discussions, issues, and pull requests.

---

## How Can I Contribute?

You can contribute in multiple ways:
* **Reporting Bugs**: Open an issue describing the bug, reproduction steps, expected vs. actual behavior, and environment details (OS, Python version, GPU/CPU).
* **Suggesting Features**: Propose new ideas such as alternative diffusion schedulers, new simulation worlds/URDFs, camera sensor augmentations, or deployment hardware interfaces.
* **Improving Documentation**: Fix typos, add docstrings, or write tutorials and walkthroughs.
* **Submitting Pull Requests**: Implement bug fixes, optimize performance, or add new capabilities.

---

## Development Environment Setup

### 1. Prerequisites
* Python 3.8 to 3.12
* PyTorch (with CUDA support recommended for GPU acceleration)
* Git

### 2. Fork and Clone the Repository
```bash
git clone https://github.com/LazyKaddu/drone-diffusion-brain.git
cd drone-diffusion-brain
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the package in editable mode along with developer tools:
```bash
pip install -e ".[dev]"
```

---

## Project Architecture

Familiarize yourself with the core modules before contributing:

```text
drone-diffusion-brain/
├── configs/
│   └── default.yaml          # Global Hydra / OmegaConf pipeline configuration
├── data/
│   └── expert_trajectories.zarr # Chunked multimodal dataset storage
├── dataset/
│   ├── collect_expert.py     # Analytic PID & raycast obstacle-avoiding data collector
│   ├── zarr_dataset.py       # Sliding-window sequence dataset & DataLoader factory
│   └── visualize_zarr.py     # Interactive Matplotlib trajectory and telemetry visualizer
├── models/
│   ├── encoder.py            # Kinematic MLP & Vision ResNet-18 conditioning encoders
│   ├── unet1d.py             # 1D Temporal U-Net backbone with FiLM conditioning
│   └── policy.py             # DDPMScheduler / DDIMScheduler diffusion policy interface
├── tests/                    # Pytest test suite covering encoders, policy, datasets, etc.
├── train.py                  # Training loop with EMA caching and validation splits
└── deploy.py                 # Real-time closed-loop receding horizon control loop
```

---

## Development Workflow

### 1. Create a Topic Branch
Always branch off the latest `main` branch with a clear prefix:
```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Implement Your Changes
* Keep pull requests focused on a single topic or bug fix.
* Maintain backward compatibility with existing configs whenever possible.
* If modifying hyperparameters or architecture dimensions, update `configs/default.yaml` accordingly.

### 3. Test Your Changes
Run the unit test suite before opening a pull request:
```bash
pytest tests/
```

To run a specific test module:
```bash
pytest tests/test_collector_utils.py
pytest tests/test_dataset.py
pytest tests/test_policy.py
```

### 4. Validate Data & Models
* If updating the data collection pipeline:
  ```bash
  python dataset/collect_expert.py --config configs/default.yaml
  ```
* If testing the interactive visualizer:
  ```bash
  python dataset/visualize_zarr.py
  ```

---

## Code Quality & Testing

To maintain a clean and maintainable codebase:
* **Style**: Follow PEP 8 guidelines. Use clear, self-explanatory variable and function names.
* **Typing**: Add type hints (`from typing import Tuple, List, Optional, Dict`) to function signatures.
* **Documentation**: Provide docstrings for public classes and functions explaining parameters, shapes, and return values.
* **Testing**: When introducing a new feature or fixing a bug, add a corresponding test in the `tests/` directory.

---

## Commit & Pull Request Guidelines

### Commit Messages
We recommend using [Conventional Commits](https://www.conventionalcommits.org/):
* `feat: add relative carrot observation conditioning`
* `fix: normalize motor RPMs to prevent simulation explosion`
* `docs: update deployment instructions in README`
* `test: add unit test for obstacle-free carrot generator`
* `refactor: optimize Zarr chunk caching`

### Pull Request Checklist
Before submitting your PR, ensure:
- [ ] Code builds and passes all tests (`pytest tests/`).
- [ ] New functions and classes include clear docstrings and type annotations.
- [ ] Relevant documentation or configuration files (`configs/default.yaml`) have been updated.
- [ ] Commit history is clean and descriptive.
- [ ] PR description clearly explains the problem, proposed solution, and verification steps.

---

## License

By contributing to **Drone Diffusion Brain**, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
