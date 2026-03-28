![banner](https://github.com/Xayah-Graphics/imagebed/blob/d975c69a760b3d5a27fdde9fd0e927dd21122e0b/stable-fluids.png)
# [SIGGRAPH 1999] Stable fluids. Jos Stam.
[![Arch Build](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/arch-build.yml)
[![Windows Build](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/windows-build.yml)

Modern C++ 23 / CUDA Implementation with C ABI of the paper [_Stable fluids_](https://dl.acm.org/doi/10.1145/311535.311548) by Jos Stam.

## 1. Algorithm Pipeline

#### Fields

- $\mathbf{w}$: velocity field
- $\mathbf{\rho}$: density field

### 1.1 Velocity Pipeline

$$
\mathbf{w}^n_0
\xrightarrow{\text{add force}} \mathbf{w}^n_1
\xrightarrow{\text{advect}} \mathbf{w}^n_2
\xrightarrow{\text{diffuse}} \mathbf{w}^n_3
\xrightarrow{\text{project}} \mathbf{w}^n_4
\Rightarrow{} \mathbf{w}^{n+1}_0.
$$

#### add force

$$
\mathbf{w}_1(\mathbf{x}) = \mathbf{w}_0(\mathbf{x}) + \Delta t\, f(\mathbf{x}, t)
$$

#### advect (semi-Lagrangian method)

$$
\mathbf{w}_2(\mathbf{x}) = \mathbf{w}_1(\mathbf{p}(\mathbf{x}, -\Delta t)).
$$

#### diffuse

$$
(I - \nu \Delta t \nabla^2) \mathbf{w}_3(\mathbf{x}) = \mathbf{w}_2(\mathbf{x})
$$

where $\nu$ is the viscosity coefficient.

#### project

$$
\nabla^2 q = \nabla \cdot \mathbf{w}_3
\qquad
\mathbf{w}_4 = \mathbf{w}_3 - \nabla q
$$

### 1.2 Density Pipeline

$$
\mathbf{\rho}^n_0 \xrightarrow{\text{add source}}
\mathbf{\rho}^n_1 \xrightarrow{\text{advect}}
\mathbf{\rho}^n_2 \xrightarrow{\text{diffuse}}
\mathbf{\rho}^n_3 \Rightarrow{}
\mathbf{\rho}^{n+1}_0.
$$


#### add source

$$
\mathbf{\rho}_1(\mathbf{x}) = \mathbf{\rho}_0(\mathbf{x}) + \Delta t S(\mathbf{x}, t)
$$

where $S(\mathbf{x}, t)$ is the scalar source term (e.g., injected smoke or dye).

#### advect (semi-Lagrangian method)

$$
\mathbf{\rho}_2(\mathbf{x}) = \mathbf{\rho}_1(\mathbf{p}(\mathbf{x}, -\Delta t))
$$

#### diffuse

$$
(I - \kappa \Delta t \nabla^2) \mathbf{\rho}_3(\mathbf{x}) = \mathbf{\rho}_2(\mathbf{x})
$$

where $\kappa$ is the diffusion coefficient (similar to viscosity for density dissipation).

## 2. Build Instruction

#### Build C ABI library
- CMake 4.3.0 or higher
- Ninja build system (for CXX std module support)
- A C++23 compliant compiler (tested on Arch Linux with gcc/g++ 15.2.1, Windows with MSVC 17.14.29)
- NVIDIA CUDA 13.2 or higher

```
cmake -B build -S . -G Ninja
cmake --build build --parallel
```

#### Build with Vulkan visualizer

A built-in Vulkan visualizer is provided. To enable it, you need to install latest Vulkan SDK.

- Vulkan SDK 1.4 or higher.

```
cmake -B build -S . -G Ninja -DSTABLE_FLUIDS_BUILD_VULKAN_APP=ON
cmake --build build --parallel
```
