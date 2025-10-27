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

Later GCC version (GCC version 9 or later) is needed for the build of GROMACS. For Ubuntu 24.04.1 LTS, you can use `sudo apt-get install build-essential` command in your terminal directly (GCC 13 is the default version for Ubuntu 24.04.1 LTS). For earlier systems you may need to update the version manually.
