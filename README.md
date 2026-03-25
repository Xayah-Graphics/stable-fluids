# [SIGGRAPH 1999] Stable fluids. Jos Stam.

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
(I - \nu \,\Delta t\, \nabla^2)\, \mathbf{w}_3(\mathbf{x}) = \mathbf{w}_2(\mathbf{x})
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
\mathbf{\rho}_1(\mathbf{x}) = \mathbf{\rho}_0(\mathbf{x}) + \Delta t\, S(\mathbf{x}, t)
$$

where $S(\mathbf{x}, t)$ is the scalar source term (e.g., injected smoke or dye).

#### advect (semi-Lagrangian method)

$$
\mathbf{\rho}_2(\mathbf{x}) = \mathbf{\rho}_1(\mathbf{p}(\mathbf{x}, -\Delta t))
$$

#### diffuse

$$
(I - \kappa \,\Delta t\, \nabla^2)\, \mathbf{\rho}_3(\mathbf{x}) = \mathbf{\rho}_2(\mathbf{x})
$$

where $\kappa$ is the diffusion coefficient (similar to viscosity for density dissipation).

# 3. Implementation Details

## 3.1 Multi-grid solver for diffusion and projection

Before discussing the exact details of the whole implementation pipeline, we first need to understand how to solve the
sparse linear systems within linear time ($\mathcal{O}(N)$ overall).

### Problem formulation: Sparse linear systems

The diffusion and projection steps both lead to large sparse linear systems of the form
$$A \mathbf{x} = \mathbf{b},$$
where \(A\) is a sparse matrix arising from the discretization of an elliptic operator. In our case, this is typically:

- a Poisson operator for the pressure projection step, or
- a Helmholtz-type operator for the implicit diffusion step.

Here, \(\mathbf{x}\) is the unknown solution vector and \(\mathbf{b}\) is the known right-hand side.

To solve these systems efficiently, several iterative methods are commonly used, such as Jacobi, Gauss-Seidel, Conjugate
Gradient, and multigrid. Their asymptotic costs are summarized below.

| Method             | Complexity                       | Note                                                            |
|--------------------|----------------------------------|-----------------------------------------------------------------|
| Jacobi             | \(\mathcal{O}(N)\) per iteration | Simple and parallel-friendly, but converges slowly              |
| Gauss-Seidel       | \(\mathcal{O}(N)\) per iteration | Usually converges faster than Jacobi, but still slow overall    |
| Conjugate Gradient | \(\mathcal{O}(N)\) per iteration | For SPD systems; total cost depends on \(\kappa\) and tolerance |
| Multigrid          | Near-\(\mathcal{O}(N)\) overall  | Near-optimal for Poisson/Helmholtz-type elliptic problems       |

### Multi-grid method

