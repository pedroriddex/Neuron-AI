# Self-Hosted Mac Mini Runner (MLX)

This directory contains documentation for configuring the **Neuron-CI Metal/MLX** runner
used in Sprint 0 parallel task **T-016**.

## Hardware
* Apple Mac Mini M2 Pro (32 GB RAM)
* macOS 13 or newer
* Xcode Command-Line Tools installed

## Setup Steps
1. **Install Homebrew tools**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.11 git wget llvm cmake pkg-config
   ```
2. **Install MLX & dependencies**
   ```bash
   python3.11 -m pip install --upgrade pip
   python3.11 -m pip install mlx-foundation-python
   # Verify
   python3.11 - <<'PY'
   import mlx
   print("MLX ready →", mlx.__version__)
   PY
   ```
3. **Create runner user** (optional)
   ```bash
   sudo sysadminctl -addUser ci -password "<pw>" -home /Users/ci -admin
   ```
4. **Configure GitHub self-hosted runner**
   *Generate token & labels (`mac-mlx`, `self-hosted`) in GitHub → Settings → Actions → Runners.*
   ```bash
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-osx-x64-2.317.0.tar.gz -L \
     https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-osx-x64-2.317.0.tar.gz
   tar xzf actions-runner-osx-x64-2.317.0.tar.gz
   ./config.sh --url https://github.com/<org>/<repo> --token <TOKEN> --labels mac-mlx
   ./run.sh &
   ```
5. **Persistent service** (launchctl or brew services). Example plist in `service.plist`.

## Smoke Test
Runner will automatically execute `.github/workflows/mac-smoke.yml` which imports
Tensor data on MLX and runs the Neuron unit tests in fp16/int8 modes.
