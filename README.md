# [SIGGRAPH 1999] Stable fluids. Jos Stam.

[![Arch Build](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/stable-fluids/actions/workflows/arch-build.yml)

## 1. Pipeline Overview

#### Fields

- $\mathbf{w}$: velocity field
- $\mathbf{\rho}$: density field

#### Velocity Pipeline

$$
\mathbf{w}^n_0
\xrightarrow{\text{add force}} \mathbf{w}^n_1
\xrightarrow{\text{advect}} \mathbf{w}^n_2
\xrightarrow{\text{diffuse}} \mathbf{w}^n_3
\xrightarrow{\text{project}} \mathbf{w}^n_4
\Rightarrow{} \mathbf{w}^{n+1}_0.
$$

#### Density Pipeline

$$
\mathbf{\rho}^n_0 \xrightarrow{\text{add source}}
\mathbf{\rho}^n_1 \xrightarrow{\text{advect}}
\mathbf{\rho}^n_2 \xrightarrow{\text{diffuse}}
\mathbf{\rho}^n_3 \Rightarrow{}
\mathbf{\rho}^{n+1}_0.
$$

## 2. Algorithm Analysis

### 2.1 Velocity Pipeline

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

### 2.2 Density Pipeline

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
