<!--
 * @Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
 * @Date: 2025-11-07 17:15:19
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2025-11-14 03:12:42
 * Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
-->
# Getting started

The code of data simulation is implemented based on [this repository](https://github.com/JMLab-tifrh/ecoli_finer) therefore [GROMACS 2019.6](https://manual.gromacs.org/2019-current/index.html) or earlier is needed.

This project runs in a Docker container. You can just clone [the image of the container](https://hub.docker.com/repository/docker/mengxin0811/bac_chrom/general) from Docker Hub and don't need to build the environment by yourself.

Environment:

* Docker image: Ubuntu 18.04
* GCC / G++ 7.5.0
* CMake 3.4.3
* CUDA 10.6 (optional)
* OpenMPI 5.0.5 (optional)
* Python 3.13.9
