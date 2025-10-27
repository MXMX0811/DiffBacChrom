# Getting started

This project requires [GROMACS 2025.3](https://manual.gromacs.org/current/index.html) and runs in the Windows Subsystem for Linux (WSL).

## Install Ubuntu 24.04.1 LTS on WSL

[WSL Documentation](https://learn.microsoft.com/en-us/windows/wsl/install-manual)

### Enable the Developer Mode in Windows

Go to Settings > Privacy & security > For developers on some versions and toggle the Developer Mode switch on.

### Enable the Windows Subsystem for Linux

Open PowerShell as Administrator (Start menu > PowerShell > right-click > Run as Administrator) and enter this command:

```Powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

### Enable Virtual Machine feature

Open PowerShell as Administrator and run:

```Powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**Restart your machine.**

### Download the Linux kernel update package

Download the package:

* [WSL2 Linux kernel update package for x64 machines](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

Double click and install the package.

Open PowerShell and run this command to set WSL 2 as the default version when installing a new Linux distribution:

```powershell
wsl --set-default-version 2
```

### Install Ubuntu 24.04.1 LTS using Microsoft Store

You can select your favorite Linux distribution on Microsoft Store. Here we use Ubuntu 24.04.1 LTS.

After the installation, a console window will open and you should create your user account and password for the Linux distribution.

## Environment dependencies

### GCC / G++

Later GCC version (GCC version 9 or later) is needed for the build of GROMACS. For Ubuntu 24.04.1 LTS, you can use `apt-get` command in your terminal directly (default version GCC 13 for Ubuntu 24.04.1 LTS).

```powershell
sudo apt-get update
sudo apt-get install gcc g++
```

Check the version:

```powershell
gcc -v
g++ -v
```

For earlier systems you may need to update the version manually.

### CMake

GROMACS builds with the CMake build system, requiring at least version 3.28. Here we use CMake 4.0.2.

OpenSSL package is needed:

```powershell
sudo apt-get install libssl-dev
```

Then download and depackage the source code of CMake:

```powershell
wget https://github.com/Kitware/CMake/releases/download/v4.0.2/cmake-4.0.2.tar.gz
tar -zxvf cmake-4.0.2.tar.gz
```

Enter the extracted folder:

```powershell
cd cmake-4.0.2
./bootstrap
make
sudo make install
```

Add environment variables in the file `~/.bashrc` (PATH=your installation path):

```powershell
export PATH="/usr/local/cmake/bin:$PATH"
```

You can use a Windows text editor to access and modify files in the WSL file system directly from Windows. If you prefer to use Vim instead, type the following command in the terminal:

```powershell
vim ~/.bashrc
```

Now you can see the file contents in the terminal window. Scroll to the end of the file and press `i` to enter insert mode, then type: `export PATH="/usr/local/cmake/bin:$PATH"`. Press `Esc` to exit insert mode. Then type `:wq` to save and quit the file.

Once you have modified `~/.bashrc`, you should update it by:

```powershell
source ~/.bashrc
```

Check the version:

```powershell
cmake -version
```

### CUDA (optional)

Here we use CUDA for GPU acceleration, therefore we should install CUDA ToolKit before build GROMACS. You should check the capabilty of your GPU driver version [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). We use CUDA 12.6 in this project. You can describe your target platform and download the software [here](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network).

Use network installer:

```powershell
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

Add environment variables:

```powershell
vim ~/.bashrc
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
source ~/.bashrc
```

Check the version:

```powershell
nvidia-smi
nvcc -V
```

### OpenMPI (optional)

You can select the OpenMPI version [here](https://www.open-mpi.org/software/ompi/v5.0/). Here we use OpenMPI 5.0.5.

```powershell
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
tar zxf openmpi-5.0.5.tar.gz
cd openmpi-5.0.5
./configure --prefix=/usr/local/openmpi
make
make install
```

Add environment variables:

```powershell
vim ~/.bashrc
MPI_HOME=/usr/local/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH
source ~/.bashrc
```

### Conda

We mainly use Python for this project. Here we use MiniConda as the Python environment. Miniconda is a free, miniature installation of Anaconda Distribution.

```powershell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

You should press `Enter` to review Miniconda’s End User License Agreement (EULA). 

Then you should type `yes` to agree the EULA.

Press `Enter` to accept the default install location (`PREFIX=/Users/<USER>/miniconda3`) or give your own path.

Then choose the initialization option, recommand `yes` (initialize conda whenever you open a new shell and to recognize conda commands automatically). You can use `conda config --set auto_activate_base false` to disable it later.

Create a conda environment for the project:

```powershell
conda create --name bac_chrom python=3.13.9
```

Activate the environment:

```powershell
conda activate bac_chrom
```

Deactivate the environment:

```powershell
conda deactivate
```

The GROMACS build requires the Python packages `MDAnalysis` and `groio`. We first need to install `pip` to allow the installation of other packages. First activate the environment:

```powershell
conda activate bac_chrom
```

Install `pip`:

```powershell
conda install pip
```

Install `MDAnalysis` and `groio`:

```powershell
pip install MDAnalysis groio
```
