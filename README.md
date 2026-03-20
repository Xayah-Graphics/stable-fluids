# Stable Fluids

## Pipeline Overview

For an incompressible velocity field $\mathbf{u}$, Stam writes

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{P}\left( -(\mathbf{u} \cdot \nabla)\mathbf{u} + \nu \nabla^2 \mathbf{u} + \mathbf{f} \right),
$$

where $\mathbf{P}$ denotes projection onto divergence-free fields.

One step is split into

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

This repository follows the same split. The only difference is that `add force` is handled outside `stable_fluids_step_cuda`, so the CUDA step starts from $\mathbf{w}_1$.

The discrete fields are:

- `density`: $(nx, ny, nz)$
- `velocity_x`: $(nx + 1, ny, nz)$
- `velocity_y`: $(nx, ny + 1, nz)$
- `velocity_z`: $(nx, ny, nz + 1)$

Velocity uses a MAC staggered grid. Density is cell-centered.

## Method Details

### 1. Advection

#### Theory

The non-linear term is treated by the method of characteristics. For

$$
\frac{\partial a}{\partial t} = - \mathbf{u} \cdot \nabla a,
$$

the value at the new time is obtained by tracing backward:

$$
a(\mathbf{x}, t + \Delta t) = a(\mathbf{p}(\mathbf{x}, -\Delta t), t).
$$

In the CUDA implementation, the trace is carried out in grid-index coordinates. If $h$ is the cell size, then

$$
\mathbf{x}_d = \mathbf{x} - \frac{\Delta t}{h}\mathbf{u}(\mathbf{x}).
$$

Because the velocity is staggered, the vector field is reconstructed at an arbitrary point by

$$
\mathbf{u}(x,y,z) =
\begin{bmatrix}
u(x, y - 0.5, z - 0.5) \\
v(x - 0.5, y, z - 0.5) \\
w(x - 0.5, y - 0.5, z)
\end{bmatrix}.
$$

For the three face-centered components, the characteristic origins are

$$
\mathbf{x}_u = (i, j + 0.5, k + 0.5),
$$

$$
\mathbf{x}_v = (i + 0.5, j, k + 0.5),
$$

$$
\mathbf{x}_w = (i + 0.5, j + 0.5, k).
$$

For the scalar density, the characteristic origin is

$$
\mathbf{x}_\rho = (i + 0.5, j + 0.5, k + 0.5).
$$

Thus the advection stage is nothing more than backward tracing followed by interpolation on the old field.

#### CUDA Implementation

The implementation uses clamped trilinear interpolation and a MAC-to-collocated reconstruction of velocity. The key code is:

```cpp
__device__ float sample_grid(
    const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    const float px = clampf(x, 0.0f, static_cast<float>(nx - 1));
    const float py = clampf(y, 0.0f, static_cast<float>(ny - 1));
    const float pz = clampf(z, 0.0f, static_cast<float>(nz - 1));
    const int i0 = static_cast<int>(floorf(px));
    const int j0 = static_cast<int>(floorf(py));
    const int k0 = static_cast<int>(floorf(pz));
    const int i1 = min(i0 + 1, nx - 1);
    const int j1 = min(j0 + 1, ny - 1);
    const int k1 = min(k0 + 1, nz - 1);
    const float tx = px - static_cast<float>(i0);
    const float ty = py - static_cast<float>(j0);
    const float tz = pz - static_cast<float>(k0);
    const float c000 = fetch_clamped(field, i0, j0, k0, nx, ny, nz);
    const float c100 = fetch_clamped(field, i1, j0, k0, nx, ny, nz);
    const float c010 = fetch_clamped(field, i0, j1, k0, nx, ny, nz);
    const float c110 = fetch_clamped(field, i1, j1, k0, nx, ny, nz);
    const float c001 = fetch_clamped(field, i0, j0, k1, nx, ny, nz);
    const float c101 = fetch_clamped(field, i1, j0, k1, nx, ny, nz);
    const float c011 = fetch_clamped(field, i0, j1, k1, nx, ny, nz);
    const float c111 = fetch_clamped(field, i1, j1, k1, nx, ny, nz);
    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0 = c00 + (c10 - c00) * ty;
    const float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

__device__ float3 sample_velocity(
    const float* const u, const float* const v, const float* const w, const float3 p, const int nx, const int ny, const int nz) {
    return make_float3(
        sample_u(u, p.x, p.y - 0.5f, p.z - 0.5f, nx, ny, nz),
        sample_v(v, p.x - 0.5f, p.y, p.z - 0.5f, nx, ny, nz),
        sample_w(w, p.x - 0.5f, p.y - 0.5f, p.z, nx, ny, nz));
}
```

and the four advection kernels:

```cpp
__global__ void advect_u_kernel(...);
__global__ void advect_v_kernel(...);
__global__ void advect_w_kernel(...);
__global__ void advect_scalar_kernel(...);
```

Operationally:

- the current velocity field is first copied into the previous-velocity buffers,
- `advect_u_kernel`, `advect_v_kernel`, and `advect_w_kernel` compute $\mathbf{w}_2$ from $\mathbf{w}_1$,
- `advect_scalar_kernel` later advects density with the projected velocity field.

### 2. Diffusion

#### Theory

The diffusion step solves

$$
\frac{\partial \mathbf{w}_2}{\partial t} = \nu \nabla^2 \mathbf{w}_2.
$$

Backward Euler gives

$$
(I - \nu \Delta t \nabla^2)\mathbf{w}_3 = \mathbf{w}_2.
$$

On the regular grid, one relaxation update is

$$
q_{i,j,k}^{m+1} =
\frac{
q_{i,j,k}^{*}
+ a \left(
q_{i-1,j,k}^{m} + q_{i+1,j,k}^{m}
+ q_{i,j-1,k}^{m} + q_{i,j+1,k}^{m}
+ q_{i,j,k-1}^{m} + q_{i,j,k+1}^{m}
\right)
}{
1 + 6a
},
$$

with

$$
a = \frac{\nu \Delta t}{h^2}.
$$

The same formula is used later for scalar diffusion, replacing $\nu$ by the scalar diffusion coefficient.

#### CUDA Implementation

The CUDA backend uses red-black Gauss-Seidel:

```cpp
__global__ void diffuse_grid_kernel(float* const dst, const float* const src, const int nx, const int ny, const int nz, const float alpha, const float denom, const int parity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    if (((i + j + k) & 1) != parity) return;
    const float center = src[index_3d(i, j, k, nx, ny)];
    const float sum = fetch_clamped(dst, i - 1, j, k, nx, ny, nz) + fetch_clamped(dst, i + 1, j, k, nx, ny, nz) +
                      fetch_clamped(dst, i, j - 1, k, nx, ny, nz) + fetch_clamped(dst, i, j + 1, k, nx, ny, nz) +
                      fetch_clamped(dst, i, j, k - 1, nx, ny, nz) + fetch_clamped(dst, i, j, k + 1, nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny)] = (center + alpha * sum) / denom;
}
```

For velocity diffusion:

- `src` is the advected velocity stored in the temporary velocity buffers,
- `dst` is the persistent velocity field,
- the solver alternates parity `0` and parity `1`,
- the boundary kernels are reapplied after each sweep.

For density diffusion:

- `src` is `temporary_density`,
- `dst` is `density`,
- the same relaxation kernel is reused.

### 3. Projection

#### Theory

The projection step is defined by

$$
\mathbf{w} = \mathbf{u} + \nabla q,
$$

where $\mathbf{u}$ is divergence-free. Taking divergence gives

$$
\nabla^2 q = \nabla \cdot \mathbf{w}.
$$

The projected field is then

$$
\mathbf{u} = \mathbf{P}\mathbf{w} = \mathbf{w} - \nabla q.
$$

For the MAC grid, the discrete divergence is

$$
d_{i,j,k} =
\frac{
u_{i+1,j,k} - u_{i,j,k}
+ v_{i,j+1,k} - v_{i,j,k}
+ w_{i,j,k+1} - w_{i,j,k}
}{h}.
$$

The Poisson equation is solved iteratively by

$$
q_{i,j,k}^{m+1} =
\frac{
q_{i-1,j,k} + q_{i+1,j,k}
+ q_{i,j-1,k} + q_{i,j+1,k}
+ q_{i,j,k-1} + q_{i,j,k+1}
- h^2 d_{i,j,k}
}{6}.
$$

Finally, the gradient of $q$ is subtracted componentwise on the staggered faces.

#### CUDA Implementation

The implementation is carried by five kernels:

```cpp
__global__ void compute_divergence_kernel(...);
__global__ void pressure_rbgs_kernel(...);
__global__ void subtract_gradient_u_kernel(...);
__global__ void subtract_gradient_v_kernel(...);
__global__ void subtract_gradient_w_kernel(...);
```

Their concrete code is:

```cpp
__global__ void compute_divergence_kernel(float* const divergence, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    divergence[index_3d(i, j, k, nx, ny)] =
        ((u[index_3d(i + 1, j, k, nx + 1, ny)] - u[index_3d(i, j, k, nx + 1, ny)]) +
         (v[index_3d(i, j + 1, k, nx, ny + 1)] - v[index_3d(i, j, k, nx, ny + 1)]) +
         (w[index_3d(i, j, k + 1, nx, ny)] - w[index_3d(i, j, k, nx, ny)])) * inv_h;
}

__global__ void pressure_rbgs_kernel(float* const pressure, const float* const divergence, const int nx, const int ny, const int nz, const float h, const int parity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    if (((i + j + k) & 1) != parity) return;
    const int im1 = clampi(i - 1, 0, nx - 1);
    const int ip1 = clampi(i + 1, 0, nx - 1);
    const int jm1 = clampi(j - 1, 0, ny - 1);
    const int jp1 = clampi(j + 1, 0, ny - 1);
    const int km1 = clampi(k - 1, 0, nz - 1);
    const int kp1 = clampi(k + 1, 0, nz - 1);
    const float sum = pressure[index_3d(im1, j, k, nx, ny)] + pressure[index_3d(ip1, j, k, nx, ny)] +
                      pressure[index_3d(i, jm1, k, nx, ny)] + pressure[index_3d(i, jp1, k, nx, ny)] +
                      pressure[index_3d(i, j, km1, nx, ny)] + pressure[index_3d(i, j, kp1, nx, ny)];
    pressure[index_3d(i, j, k, nx, ny)] =
        (sum - divergence[index_3d(i, j, k, nx, ny)] * h * h) / 6.0f;
}

__global__ void subtract_gradient_u_kernel(float* const u, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i == 0 || i >= nx || j >= ny || k >= nz) return;
    u[index_3d(i, j, k, nx + 1, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) * inv_h;
}

__global__ void subtract_gradient_v_kernel(float* const v, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j == 0 || j >= ny || k >= nz) return;
    v[index_3d(i, j, k, nx, ny + 1)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) * inv_h;
}

__global__ void subtract_gradient_w_kernel(float* const w, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k == 0 || k >= nz) return;
    w[index_3d(i, j, k, nx, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) * inv_h;
}
```

In the step routine:

1. `temporary_pressure` is cleared,
2. `temporary_divergence` is assembled,
3. the pressure equation is iterated by red-black sweeps,
4. the pressure gradient is subtracted from the velocity field,
5. wall conditions are enforced again.

This is the transition

$$
\mathbf{w}_3 \rightarrow \mathbf{w}_4.
$$

### 4. Scalar Transport

#### Theory

The paper treats transported substances with

$$
\frac{\partial a}{\partial t} = - \mathbf{u} \cdot \nabla a + \kappa_a \nabla^2 a - \alpha_a a + S_a.
$$

This implementation uses only the advection-diffusion part for smoke density:

$$
\frac{\partial \rho}{\partial t} = - \mathbf{u} \cdot \nabla \rho + \kappa \nabla^2 \rho.
$$

The advection part again uses backward characteristics:

$$
\rho^*(\mathbf{x}, t + \Delta t) = \rho(\mathbf{p}(\mathbf{x}, -\Delta t), t),
$$

and the diffusion part uses

$$
(I - \kappa \Delta t \nabla^2)\rho^{n+1} = \rho^*.
$$

#### CUDA Implementation

The scalar advection kernel is:

```cpp
__global__ void advect_scalar_kernel(float* const dst, const float* const src, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float dt_over_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const float3 p = make_float3(static_cast<float>(i) + 0.5f, static_cast<float>(j) + 0.5f, static_cast<float>(k) + 0.5f);
    const float3 vel = sample_velocity(u, v, w, p, nx, ny, nz);
    const float3 back = clamp_domain(make_float3(p.x - dt_over_h * vel.x, p.y - dt_over_h * vel.y, p.z - dt_over_h * vel.z), nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny)] = fmaxf(0.0f, sample_scalar(src, back.x - 0.5f, back.y - 0.5f, back.z - 0.5f, nx, ny, nz));
}
```

The scalar diffusion step reuses `diffuse_grid_kernel`.

Operationally:

1. `density` is copied to `temporary_previous_density`,
2. `advect_scalar_kernel` writes the advected result into `temporary_density`,
3. `diffuse_grid_kernel` relaxes from `temporary_density` into `density`.

## Reproduction Checklist

To reproduce the CUDA solver:

1. store velocity on a MAC grid,
2. reconstruct $\mathbf{u}$ at arbitrary points by half-cell shifts,
3. advect by backward tracing,
4. diffuse by red-black Gauss-Seidel,
5. project by solving the Poisson equation for $q$ and subtracting $\nabla q$,
6. advect and diffuse density with the projected velocity,
7. enforce zero normal velocity at the box walls,
8. keep the stage order

$$
\mathbf{w}_1 \rightarrow \mathbf{w}_2 \rightarrow \mathbf{w}_3 \rightarrow \mathbf{w}_4
$$

for velocity, followed by scalar advection and scalar diffusion.
