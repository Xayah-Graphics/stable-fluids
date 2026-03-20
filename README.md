# Stable Fluids

## 1. Pipeline Overview

$$
\text{Algorithms} + \text{Data Structures} = \text{Program}
$$

### 1.1 Data Structure: `Fields`

For a typical smoke simulation, the `Data Structures` we manipulate are several `Fields`, which are discrete
representations of continuous physical quantities defined on the simulation domain.
`Fields` may be defined in a any Euclidean space ($\mathbb{R}^2$, $\mathbb{R}^3$, etc.).

Typically, we may roughly categorize these `Fields` into 2 groups:

1. `Scalar Fields`: each cell stores a single value, e.g. `smoke density`, `temperature`, `pressure`, etc.
2. `Vector Fields`: each cell stores a vector, e.g. `velocity`.

For better understanding of multiple different `Fields`, please refer to [Dense Fields](TODO)

### 1.2 Algorithms

The `Algorithms` are the numerical methods that operate on the `Fields` to evolve the simulation over time. In Computer
Graphics Literature, we usually refer to these `Algorithms` as `Solvers`. For example, the `Stable Fluids` algorithm is
a `Solver`.

`Stable Fluids` is the very first `Solver` that can stably simulate turbulent fluids with large time steps, which could
be seen as the initial milestone in the history of fluid simulation for graphics.

The primary pipeline of `Stable Fluids` consists of the following 4 stages:

$$
\mathbf{w}_0
\xrightarrow{\text{add force}}
\mathbf{w}_1
\xrightarrow{\text{advect}}
\mathbf{w}_2
\xrightarrow{\text{diffuse}}
\mathbf{w}_3
\xrightarrow{\text{project}}
\mathbf{w}_4.
$$

here, $\mathbf{w}$ denotes the velocity field, and the subscript indicates the stage. The stages are:

1. `add force`: add external forces to the velocity field, e.g. gravity, buoyancy, etc.
2. `advect`: transport the velocity field by itself, i.e. self-advection.
3. `diffuse`: model the viscous diffusion of the velocity field.
4. `project`: enforce the incompressibility constraint by projecting the velocity field onto the space of
   divergence-free fields.
