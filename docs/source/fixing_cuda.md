---
orphan: true
---

# Fixing CUDA for CMake

CMake's `find_package(CUDAToolkit)` expects the CUDA Toolkit at
`/usr/local/cuda/` with a standard directory layout.  If your system
has CUDA installed elsewhere (e.g. via the NVIDIA HPC SDK or Ubuntu's
`nvidia-cuda-toolkit` package), CMake may fail with errors like:

```
Imported target "CUDA::cudart" includes non-existent path
```

## Diagnosing your setup

```bash
# Where is nvcc?
which nvcc

# What CUDA installations exist?
ls /usr/local/cuda*/bin/nvcc 2>/dev/null          # standard install
dpkg -l | grep nvidia-cuda-toolkit                 # Ubuntu package
ls /opt/nvidia/hpc_sdk/*/compilers/bin/nvcc 2>/dev/null  # HPC SDK
```

## Option A: Symlink (no downtime — recommended if HPC SDK is installed)

If you have the HPC SDK with CUDA 13.0 at the default path:

```bash
# Create the symlink that CMake expects
sudo ln -s /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/cuda/13.0 /usr/local/cuda

# Verify
ls /usr/local/cuda/bin/nvcc
ls /usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h
ls /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
```

Optionally remove the older Ubuntu CUDA 12.0 package to avoid confusion:

```bash
sudo apt remove nvidia-cuda-toolkit
```

Then add to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Option B: Clean install from NVIDIA's repo

Use this if you don't have a working CUDA install, or want a fresh start.

### Step 1: Remove existing CUDA installations

```bash
# Remove Ubuntu's packaged CUDA toolkit (12.0)
sudo apt remove --purge nvidia-cuda-toolkit nvidia-cuda-toolkit-doc
sudo apt autoremove

# Remove HPC SDK CUDA (if installed and you don't need it)
# sudo rm -rf /opt/nvidia/hpc_sdk

# Remove any existing /usr/local/cuda symlinks
sudo rm -f /usr/local/cuda
```

**Do NOT remove the NVIDIA driver** (`nvidia-driver-*`) — that's separate
from the CUDA toolkit and removing it will kill your display.

### Step 2: Add NVIDIA's official package repo

```bash
# For Ubuntu 24.04 x86_64:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

For other Ubuntu versions, replace `ubuntu2404` with your version
(`ubuntu2204`, `ubuntu2004`, etc.).  Check
<https://developer.nvidia.com/cuda-downloads> for the exact URL.

### Step 3: Install the CUDA Toolkit

```bash
# Install just the toolkit (compiler + libraries), NOT the driver
sudo apt install cuda-toolkit-13-0
```

This installs to `/usr/local/cuda-13.0/` and creates a symlink at
`/usr/local/cuda/`.

### Step 4: Set up your PATH

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### Step 5: Verify

```bash
nvcc --version
# Should show: Cuda compilation tools, release 13.0, ...

python3 -c "
import subprocess, sys
r = subprocess.run(['cmake', '-S', '.', '-B', '/tmp/cuda-test', '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Debug'],
                   capture_output=True, text=True, cwd='$(pwd)')
for line in r.stdout.splitlines():
    if 'CUDA' in line or 'cuda' in line.lower():
        print(line)
"
# Should show: "-- CUDA acceleration turned ON."
```

## Verifying the tessera build

```bash
# Clean rebuild
rm -rf cmake-build-debug
pip install -e .

# Run tests
pytest tests/ -x -v

# Run the Regge solver example
python examples/curvature_slice_gif.py --n-simplices 200 --seed 7 \
    --max-iters 10 --save point_mass.gif
```

## Troubleshooting

**"CUDA requested but nvcc not found"**
→ nvcc is not on your PATH.  Run `which nvcc` and fix your PATH.

**"Imported target CUDA::cudart includes non-existent path"**
→ CMake found CUDA metadata but the include directory doesn't exist.
  This usually means a broken or partial install.  Use Option A (symlink)
  or Option B (clean install).

**Build succeeds but `import tessera` crashes with "libcudart.so not found"**
→ Add the CUDA lib directory to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**"CUDA error: no kernel image is available for execution on the device"**
→ The CUDA code was compiled for a different GPU architecture.  Add your
  GPU's compute capability to CMakeLists.txt:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4090 = sm_89
```
