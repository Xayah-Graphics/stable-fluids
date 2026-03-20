# [SIGGRAPH 1999] Stable fluids. Jos Stam.

## 1. Pipeline Overview

#### Fields

- $\mathbf{w}$: velocity field
- $\mathbf{d}$: density field

#### Velocity Pipeline

$$
\mathbf{w}^n_0
\xrightarrow{\text{add force}}
\mathbf{w}^n_1
\xrightarrow{\text{advect}}
\mathbf{w}^n_2
\xrightarrow{\text{diffuse}}
\mathbf{w}^n_3
\xrightarrow{\text{project}}
\mathbf{w}^n_4
\xrightarrow{}
\mathbf{w}^{n+1}_0.
$$

#### Density Pipeline

$$
\mathbf{\rho}^n_0
\xrightarrow{\text{add source}}
\mathbf{\rho}^n_1
\xrightarrow{\text{advect}}
\mathbf{\rho}^n_2
\xrightarrow{\text{diffuse}}
\mathbf{\rho}^n_3
\xrightarrow{}
\mathbf{\rho}^{n+1}_0.
$$

## 2. Algorithm Analysis

### 2.1 Velocity Pipeline

#### add force

