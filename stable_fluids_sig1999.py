from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from PIL import Image


@triton.jit
def _flatten_index(x, y, z, nx, ny):
    return x + nx * (y + ny * z)


@triton.jit
def _sample_trilinear(field_ptr, x, y, z, nx, ny, nz):
    x = tl.maximum(0.0, tl.minimum(x, tl.cast(nx - 1, tl.float32)))
    y = tl.maximum(0.0, tl.minimum(y, tl.cast(ny - 1, tl.float32)))
    z = tl.maximum(0.0, tl.minimum(z, tl.cast(nz - 1, tl.float32)))

    x0 = tl.cast(tl.floor(x), tl.int32)
    y0 = tl.cast(tl.floor(y), tl.int32)
    z0 = tl.cast(tl.floor(z), tl.int32)

    x1 = tl.minimum(x0 + 1, nx - 1)
    y1 = tl.minimum(y0 + 1, ny - 1)
    z1 = tl.minimum(z0 + 1, nz - 1)

    fx = x - tl.cast(x0, tl.float32)
    fy = y - tl.cast(y0, tl.float32)
    fz = z - tl.cast(z0, tl.float32)

    c000 = tl.load(field_ptr + _flatten_index(x0, y0, z0, nx, ny))
    c100 = tl.load(field_ptr + _flatten_index(x1, y0, z0, nx, ny))
    c010 = tl.load(field_ptr + _flatten_index(x0, y1, z0, nx, ny))
    c110 = tl.load(field_ptr + _flatten_index(x1, y1, z0, nx, ny))
    c001 = tl.load(field_ptr + _flatten_index(x0, y0, z1, nx, ny))
    c101 = tl.load(field_ptr + _flatten_index(x1, y0, z1, nx, ny))
    c011 = tl.load(field_ptr + _flatten_index(x0, y1, z1, nx, ny))
    c111 = tl.load(field_ptr + _flatten_index(x1, y1, z1, nx, ny))

    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


@triton.jit
def advect_kernel(
    src_ptr,
    velx_ptr,
    vely_ptr,
    velz_ptr,
    dst_ptr,
    n_elements,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    dt,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = offs % nx
    yz = offs // nx
    y = yz % ny
    z = yz // ny

    xf = tl.cast(x, tl.float32)
    yf = tl.cast(y, tl.float32)
    zf = tl.cast(z, tl.float32)

    vx = tl.load(velx_ptr + offs, mask=mask, other=0.0)
    vy = tl.load(vely_ptr + offs, mask=mask, other=0.0)
    vz = tl.load(velz_ptr + offs, mask=mask, other=0.0)

    px = tl.maximum(1.0, tl.minimum(xf - dt * vx, tl.cast(nx - 2, tl.float32)))
    py = tl.maximum(1.0, tl.minimum(yf - dt * vy, tl.cast(ny - 2, tl.float32)))
    pz = tl.maximum(1.0, tl.minimum(zf - dt * vz, tl.cast(nz - 2, tl.float32)))

    advected = _sample_trilinear(src_ptr, px, py, pz, nx, ny, nz)
    tl.store(dst_ptr + offs, advected, mask=mask)


@triton.jit
def divergence_kernel(
    velx_ptr,
    vely_ptr,
    velz_ptr,
    div_ptr,
    n_elements,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = offs % nx
    yz = offs // nx
    y = yz % ny
    z = yz // ny

    is_boundary = (x == 0) | (x == nx - 1) | (y == 0) | (y == ny - 1) | (z == 0) | (z == nz - 1)

    xm = tl.maximum(x - 1, 0)
    xp = tl.minimum(x + 1, nx - 1)
    ym = tl.maximum(y - 1, 0)
    yp = tl.minimum(y + 1, ny - 1)
    zm = tl.maximum(z - 1, 0)
    zp = tl.minimum(z + 1, nz - 1)

    ux_m = tl.load(velx_ptr + _flatten_index(xm, y, z, nx, ny), mask=mask, other=0.0)
    ux_p = tl.load(velx_ptr + _flatten_index(xp, y, z, nx, ny), mask=mask, other=0.0)
    uy_m = tl.load(vely_ptr + _flatten_index(x, ym, z, nx, ny), mask=mask, other=0.0)
    uy_p = tl.load(vely_ptr + _flatten_index(x, yp, z, nx, ny), mask=mask, other=0.0)
    uz_m = tl.load(velz_ptr + _flatten_index(x, y, zm, nx, ny), mask=mask, other=0.0)
    uz_p = tl.load(velz_ptr + _flatten_index(x, y, zp, nx, ny), mask=mask, other=0.0)

    div = 0.5 * ((ux_p - ux_m) + (uy_p - uy_m) + (uz_p - uz_m))
    div = tl.where(is_boundary, 0.0, div)
    tl.store(div_ptr + offs, div, mask=mask)


@triton.jit
def pressure_jacobi_kernel(
    pressure_ptr,
    divergence_ptr,
    pressure_out_ptr,
    n_elements,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = offs % nx
    yz = offs // nx
    y = yz % ny
    z = yz // ny

    is_boundary = (x == 0) | (x == nx - 1) | (y == 0) | (y == ny - 1) | (z == 0) | (z == nz - 1)

    xm = tl.maximum(x - 1, 0)
    xp = tl.minimum(x + 1, nx - 1)
    ym = tl.maximum(y - 1, 0)
    yp = tl.minimum(y + 1, ny - 1)
    zm = tl.maximum(z - 1, 0)
    zp = tl.minimum(z + 1, nz - 1)

    center = tl.load(pressure_ptr + offs, mask=mask, other=0.0)
    pxm = tl.load(pressure_ptr + _flatten_index(xm, y, z, nx, ny), mask=mask, other=0.0)
    pxp = tl.load(pressure_ptr + _flatten_index(xp, y, z, nx, ny), mask=mask, other=0.0)
    pym = tl.load(pressure_ptr + _flatten_index(x, ym, z, nx, ny), mask=mask, other=0.0)
    pyp = tl.load(pressure_ptr + _flatten_index(x, yp, z, nx, ny), mask=mask, other=0.0)
    pzm = tl.load(pressure_ptr + _flatten_index(x, y, zm, nx, ny), mask=mask, other=0.0)
    pzp = tl.load(pressure_ptr + _flatten_index(x, y, zp, nx, ny), mask=mask, other=0.0)
    div = tl.load(divergence_ptr + offs, mask=mask, other=0.0)

    updated = (pxm + pxp + pym + pyp + pzm + pzp - div) / 6.0
    updated = tl.where(is_boundary, center, updated)
    tl.store(pressure_out_ptr + offs, updated, mask=mask)


@triton.jit
def diffuse_jacobi_kernel(
    current_ptr,
    source_ptr,
    output_ptr,
    n_elements,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    alpha,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = offs % nx
    yz = offs // nx
    y = yz % ny
    z = yz // ny

    is_boundary = (x == 0) | (x == nx - 1) | (y == 0) | (y == ny - 1) | (z == 0) | (z == nz - 1)

    xm = tl.maximum(x - 1, 0)
    xp = tl.minimum(x + 1, nx - 1)
    ym = tl.maximum(y - 1, 0)
    yp = tl.minimum(y + 1, ny - 1)
    zm = tl.maximum(z - 1, 0)
    zp = tl.minimum(z + 1, nz - 1)

    center = tl.load(current_ptr + offs, mask=mask, other=0.0)
    source = tl.load(source_ptr + offs, mask=mask, other=0.0)

    nsum = (
        tl.load(current_ptr + _flatten_index(xm, y, z, nx, ny), mask=mask, other=0.0)
        + tl.load(current_ptr + _flatten_index(xp, y, z, nx, ny), mask=mask, other=0.0)
        + tl.load(current_ptr + _flatten_index(x, ym, z, nx, ny), mask=mask, other=0.0)
        + tl.load(current_ptr + _flatten_index(x, yp, z, nx, ny), mask=mask, other=0.0)
        + tl.load(current_ptr + _flatten_index(x, y, zm, nx, ny), mask=mask, other=0.0)
        + tl.load(current_ptr + _flatten_index(x, y, zp, nx, ny), mask=mask, other=0.0)
    )

    updated = (source + alpha * nsum) / beta
    updated = tl.where(is_boundary, center, updated)
    tl.store(output_ptr + offs, updated, mask=mask)


@triton.jit
def project_kernel(
    vel_ptr,
    pressure_ptr,
    vel_out_ptr,
    n_elements,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    axis: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = offs % nx
    yz = offs // nx
    y = yz % ny
    z = yz // ny

    is_boundary = (x == 0) | (x == nx - 1) | (y == 0) | (y == ny - 1) | (z == 0) | (z == nz - 1)

    xm = tl.maximum(x - 1, 0)
    xp = tl.minimum(x + 1, nx - 1)
    ym = tl.maximum(y - 1, 0)
    yp = tl.minimum(y + 1, ny - 1)
    zm = tl.maximum(z - 1, 0)
    zp = tl.minimum(z + 1, nz - 1)

    center = tl.load(vel_ptr + offs, mask=mask, other=0.0)
    pxm = tl.load(pressure_ptr + _flatten_index(xm, y, z, nx, ny), mask=mask, other=0.0)
    pxp = tl.load(pressure_ptr + _flatten_index(xp, y, z, nx, ny), mask=mask, other=0.0)
    pym = tl.load(pressure_ptr + _flatten_index(x, ym, z, nx, ny), mask=mask, other=0.0)
    pyp = tl.load(pressure_ptr + _flatten_index(x, yp, z, nx, ny), mask=mask, other=0.0)
    pzm = tl.load(pressure_ptr + _flatten_index(x, y, zm, nx, ny), mask=mask, other=0.0)
    pzp = tl.load(pressure_ptr + _flatten_index(x, y, zp, nx, ny), mask=mask, other=0.0)

    grad = tl.where(
        axis == 0,
        0.5 * (pxp - pxm),
        tl.where(axis == 1, 0.5 * (pyp - pym), 0.5 * (pzp - pzm)),
    )
    projected = center - grad
    projected = tl.where(is_boundary, 0.0, projected)
    tl.store(vel_out_ptr + offs, projected, mask=mask)


@dataclass
class SimulationConfig:
    nx: int = 128
    ny: int = 128
    nz: int = 128
    dt: float = 0.05
    density_diffusion: float = 1.0e-4
    temperature_diffusion: float = 2.0e-4
    viscosity: float = 1.0e-4
    pressure_iterations: int = 80
    diffusion_iterations: int = 16
    buoyancy_alpha: float = 1.35
    buoyancy_beta: float = 0.04
    density_decay: float = 0.998
    temperature_decay: float = 0.985
    source_radius: float = 10.0
    source_density: float = 5.0
    source_temperature: float = 16.0
    source_updraft: float = 0.6
    device: str = "cuda"


class StableFluids3D:
    def __init__(self, cfg: SimulationConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Triton implementation.")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.shape = (cfg.nz, cfg.ny, cfg.nx)
        self.n_elements = cfg.nx * cfg.ny * cfg.nz
        self.grid = lambda meta: (triton.cdiv(self.n_elements, meta["BLOCK_SIZE"]),)

        self.u = torch.zeros(self.shape, device=self.device, dtype=torch.float32)
        self.v = torch.zeros_like(self.u)
        self.w = torch.zeros_like(self.u)
        self.u_tmp = torch.zeros_like(self.u)
        self.v_tmp = torch.zeros_like(self.u)
        self.w_tmp = torch.zeros_like(self.u)

        self.density = torch.zeros_like(self.u)
        self.density_tmp = torch.zeros_like(self.u)
        self.temperature = torch.zeros_like(self.u)
        self.temperature_tmp = torch.zeros_like(self.u)

        self.divergence = torch.zeros_like(self.u)
        self.pressure = torch.zeros_like(self.u)
        self.pressure_tmp = torch.zeros_like(self.u)

        self._build_source_mask()

    def _build_source_mask(self) -> None:
        z, y, x = torch.meshgrid(
            torch.arange(self.cfg.nz, device=self.device, dtype=torch.float32),
            torch.arange(self.cfg.ny, device=self.device, dtype=torch.float32),
            torch.arange(self.cfg.nx, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        center_x = 0.5 * (self.cfg.nx - 1)
        center_y = 0.5 * (self.cfg.ny - 1)
        center_z = self.cfg.source_radius * 0.75
        radius2 = self.cfg.source_radius * self.cfg.source_radius
        self.source_mask = (
            (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2 <= radius2
        )

    @staticmethod
    def _scalar_boundary(field: torch.Tensor) -> None:
        field[0, :, :] = field[1, :, :]
        field[-1, :, :] = field[-2, :, :]
        field[:, 0, :] = field[:, 1, :]
        field[:, -1, :] = field[:, -2, :]
        field[:, :, 0] = field[:, :, 1]
        field[:, :, -1] = field[:, :, -2]

    @staticmethod
    def _velocity_boundary(*fields: torch.Tensor) -> None:
        for field in fields:
            field[0, :, :] = 0.0
            field[-1, :, :] = 0.0
            field[:, 0, :] = 0.0
            field[:, -1, :] = 0.0
            field[:, :, 0] = 0.0
            field[:, :, -1] = 0.0

    def _add_sources_and_forces(self) -> None:
        dt = self.cfg.dt
        mask = self.source_mask

        self.density[mask] += dt * self.cfg.source_density
        self.temperature[mask] += dt * self.cfg.source_temperature
        self.w[mask] += dt * self.cfg.source_updraft

        buoyancy = self.cfg.buoyancy_alpha * self.temperature - self.cfg.buoyancy_beta * self.density
        self.w += dt * buoyancy

        self.density.clamp_(min=0.0)
        self.temperature.clamp_(min=0.0)
        self._scalar_boundary(self.density)
        self._scalar_boundary(self.temperature)
        self._velocity_boundary(self.u, self.v, self.w)

    def _advect(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        advect_kernel[self.grid](
            src,
            self.u,
            self.v,
            self.w,
            dst,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            dt=self.cfg.dt,
            BLOCK_SIZE=256,
        )

    def _diffuse_in_place(self, field: torch.Tensor, tmp: torch.Tensor, rate: float, scalar_field: bool) -> None:
        if rate <= 0.0:
            return

        source = field.clone()
        alpha = self.cfg.dt * rate
        beta = 1.0 + 6.0 * alpha
        current = field
        scratch = tmp

        for _ in range(self.cfg.diffusion_iterations):
            diffuse_jacobi_kernel[self.grid](
                current,
                source,
                scratch,
                self.n_elements,
                nx=self.cfg.nx,
                ny=self.cfg.ny,
                nz=self.cfg.nz,
                alpha=alpha,
                beta=beta,
                BLOCK_SIZE=256,
            )
            if scalar_field:
                self._scalar_boundary(scratch)
            else:
                self._velocity_boundary(scratch)
            current, scratch = scratch, current

        if current.data_ptr() != field.data_ptr():
            field.copy_(current)

    def _compute_divergence(self) -> None:
        divergence_kernel[self.grid](
            self.u,
            self.v,
            self.w,
            self.divergence,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            BLOCK_SIZE=256,
        )
        self._scalar_boundary(self.divergence)

    def _project(self) -> None:
        self._compute_divergence()
        self.pressure.zero_()
        self.pressure_tmp.zero_()

        for _ in range(self.cfg.pressure_iterations):
            pressure_jacobi_kernel[self.grid](
                self.pressure,
                self.divergence,
                self.pressure_tmp,
                self.n_elements,
                nx=self.cfg.nx,
                ny=self.cfg.ny,
                nz=self.cfg.nz,
                BLOCK_SIZE=256,
            )
            self._scalar_boundary(self.pressure_tmp)
            self.pressure, self.pressure_tmp = self.pressure_tmp, self.pressure

        project_kernel[self.grid](
            self.u,
            self.pressure,
            self.u_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            axis=0,
            BLOCK_SIZE=256,
        )
        project_kernel[self.grid](
            self.v,
            self.pressure,
            self.v_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            axis=1,
            BLOCK_SIZE=256,
        )
        project_kernel[self.grid](
            self.w,
            self.pressure,
            self.w_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            axis=2,
            BLOCK_SIZE=256,
        )
        self.u.copy_(self.u_tmp)
        self.v.copy_(self.v_tmp)
        self.w.copy_(self.w_tmp)
        self._velocity_boundary(self.u, self.v, self.w)

    def step(self) -> None:
        self._add_sources_and_forces()

        self._diffuse_in_place(self.u, self.u_tmp, self.cfg.viscosity, scalar_field=False)
        self._diffuse_in_place(self.v, self.v_tmp, self.cfg.viscosity, scalar_field=False)
        self._diffuse_in_place(self.w, self.w_tmp, self.cfg.viscosity, scalar_field=False)
        self._project()

        u_prev = self.u.clone()
        v_prev = self.v.clone()
        w_prev = self.w.clone()

        advect_kernel[self.grid](
            u_prev,
            u_prev,
            v_prev,
            w_prev,
            self.u_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            dt=self.cfg.dt,
            BLOCK_SIZE=256,
        )
        advect_kernel[self.grid](
            v_prev,
            u_prev,
            v_prev,
            w_prev,
            self.v_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            dt=self.cfg.dt,
            BLOCK_SIZE=256,
        )
        advect_kernel[self.grid](
            w_prev,
            u_prev,
            v_prev,
            w_prev,
            self.w_tmp,
            self.n_elements,
            nx=self.cfg.nx,
            ny=self.cfg.ny,
            nz=self.cfg.nz,
            dt=self.cfg.dt,
            BLOCK_SIZE=256,
        )
        self.u.copy_(self.u_tmp)
        self.v.copy_(self.v_tmp)
        self.w.copy_(self.w_tmp)
        self._velocity_boundary(self.u, self.v, self.w)
        self._project()

        self._diffuse_in_place(self.density, self.density_tmp, self.cfg.density_diffusion, scalar_field=True)
        self._diffuse_in_place(
            self.temperature, self.temperature_tmp, self.cfg.temperature_diffusion, scalar_field=True
        )

        density_prev = self.density.clone()
        temperature_prev = self.temperature.clone()

        self._advect(density_prev, self.density_tmp)
        self._advect(temperature_prev, self.temperature_tmp)
        self.density.copy_(self.density_tmp)
        self.temperature.copy_(self.temperature_tmp)

        self.density.mul_(self.cfg.density_decay).clamp_(min=0.0)
        self.temperature.mul_(self.cfg.temperature_decay).clamp_(min=0.0)
        self._scalar_boundary(self.density)
        self._scalar_boundary(self.temperature)

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "density": self.density.detach().clone(),
            "temperature": self.temperature.detach().clone(),
            "u": self.u.detach().clone(),
            "v": self.v.detach().clone(),
            "w": self.w.detach().clone(),
            "pressure": self.pressure.detach().clone(),
        }


def format_stats(step: int, sim: StableFluids3D) -> str:
    density = sim.density
    velocity_mag = torch.sqrt(sim.u * sim.u + sim.v * sim.v + sim.w * sim.w)
    return (
        f"step={step:04d} "
        f"density[max={density.max().item():.4f}, sum={density.sum().item():.4f}] "
        f"velocity[max={velocity_mag.max().item():.4f}, mean={velocity_mag.mean().item():.4f}]"
    )


def _normalize_image(values: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    values = np.maximum(values, 0.0)
    scale = float(np.percentile(values, percentile))
    if scale <= 1.0e-8:
        scale = float(values.max()) if values.size else 1.0
    if scale <= 1.0e-8:
        scale = 1.0
    normalized = np.clip(values / scale, 0.0, 1.0)
    return normalized


def _orient_vertical(image: np.ndarray) -> np.ndarray:
    return np.flipud(image)


def _smoke_palette(intensity: np.ndarray, heat: np.ndarray) -> np.ndarray:
    base = np.stack(
        [
            0.08 + 0.92 * intensity,
            0.09 + 0.88 * intensity,
            0.12 + 0.82 * intensity,
        ],
        axis=-1,
    )
    fire = np.stack(
        [
            0.95 * heat + 0.15 * intensity,
            0.55 * heat + 0.20 * intensity,
            0.18 * heat + 0.25 * intensity,
        ],
        axis=-1,
    )
    rgb = np.clip(base + fire, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def render_visualizations(sim: StableFluids3D) -> dict[str, np.ndarray]:
    density = sim.density.detach().float().cpu().numpy()
    temperature = sim.temperature.detach().float().cpu().numpy()
    velocity_mag = torch.sqrt(sim.u * sim.u + sim.v * sim.v + sim.w * sim.w).detach().float().cpu().numpy()

    mid_z = density.shape[0] // 2
    mid_y = density.shape[1] // 2
    mid_x = density.shape[2] // 2

    density_slice = _normalize_image(density[mid_z])
    temperature_slice = _normalize_image(temperature[mid_z])
    velocity_slice = _normalize_image(velocity_mag[mid_z])

    density_rgb = _smoke_palette(density_slice, 0.35 * temperature_slice)
    temperature_rgb = _smoke_palette(0.5 * temperature_slice, temperature_slice)
    velocity_rgb = _smoke_palette(0.65 * velocity_slice, 0.25 * temperature_slice)

    xz_density = _normalize_image(density[:, mid_y, :])
    yz_density = _normalize_image(density[:, :, mid_x])
    xz_rgb = _smoke_palette(xz_density, 0.4 * _normalize_image(temperature[:, mid_y, :]))
    yz_rgb = _smoke_palette(yz_density, 0.4 * _normalize_image(temperature[:, :, mid_x]))
    xz_rgb = _orient_vertical(xz_rgb)
    yz_rgb = _orient_vertical(yz_rgb)

    density_norm = _normalize_image(density, percentile=99.8)
    temperature_norm = _normalize_image(temperature, percentile=99.8)

    alpha = np.clip(density_norm * 0.08, 0.0, 0.35)
    color = np.stack(
        [
            0.06 + 0.90 * density_norm + 0.70 * temperature_norm,
            0.08 + 0.82 * density_norm + 0.30 * temperature_norm,
            0.12 + 0.75 * density_norm + 0.08 * temperature_norm,
        ],
        axis=-1,
    )
    transmittance_front = np.cumprod(1.0 - alpha + 1.0e-4, axis=1)
    shifted_front = np.concatenate(
        [np.ones_like(transmittance_front[:, :1]), transmittance_front[:, :-1]],
        axis=1,
    )
    volume_front = np.sum(color * alpha[..., None] * shifted_front[..., None], axis=1)
    volume_front = np.clip(volume_front, 0.0, 1.0)
    volume_front = _orient_vertical((volume_front * 255.0).astype(np.uint8))

    transmittance_top = np.cumprod(1.0 - alpha + 1.0e-4, axis=0)
    shifted = np.concatenate(
        [np.ones_like(transmittance_top[:1]), transmittance_top[:-1]],
        axis=0,
    )
    volume_top = np.sum(color * alpha[..., None] * shifted[..., None], axis=0)
    volume_top = np.clip(volume_top, 0.0, 1.0)
    volume_top = (volume_top * 255.0).astype(np.uint8)

    top_density = np.max(density_norm, axis=0)
    top_heat = np.max(temperature_norm, axis=0)
    top_rgb = _smoke_palette(top_density, top_heat)
    return {
        "density_xy": density_rgb,
        "temperature_xy": temperature_rgb,
        "velocity_xy": velocity_rgb,
        "density_xz": xz_rgb,
        "density_yz": yz_rgb,
        "volume_front": volume_front,
        "volume_top": volume_top,
        "mip_top": top_rgb,
    }


def save_visualizations(sim: StableFluids3D, output_dir: Path, step: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    views = render_visualizations(sim)
    for name, image in views.items():
        path = output_dir / f"{name}_step_{step:04d}.png"
        Image.fromarray(image, mode="RGB").save(path)
        written[name] = path
    return written


def export_animation_frame(sim: StableFluids3D, frame_dir: Path, step: int, view: str) -> Path:
    frame_dir.mkdir(parents=True, exist_ok=True)
    image = render_visualizations(sim)[view]
    path = frame_dir / f"{view}_step_{step:04d}.png"
    Image.fromarray(image, mode="RGB").save(path)
    return path


def build_gif_from_frames(frame_dir: Path, output_path: Path, fps: int) -> bool:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return False

    images = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    duration_ms = max(1, int(round(1000 / max(fps, 1))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()
    return True


def build_mp4_from_frames(frame_dir: Path, output_path: Path, fps: int) -> tuple[bool, str]:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return False, "no frames found"

    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_bin is None:
        return False, "ffmpeg not found"

    pattern = str(frame_dir / "*.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(max(fps, 1)),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        return False, message
    return True, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D Stable Fluids smoke plume using Triton.")
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps to run. Overrides --sim-seconds when set.")
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep in simulation seconds.")
    parser.add_argument(
        "--sim-seconds",
        type=float,
        default=12.0,
        help="Total simulated time in seconds when --steps is not specified.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the final state as .pt.")
    parser.add_argument("--stats-every", type=int, default=20, help="Print stats every N steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for PNG slice and volume-render exports.",
    )
    parser.add_argument(
        "--export-every",
        type=int,
        default=0,
        help="Export PNG visualizations every N steps. Use 0 to disable intermediate exports.",
    )
    parser.add_argument(
        "--export-final",
        action="store_true",
        help="Export PNG visualizations for the final state.",
    )
    parser.add_argument(
        "--animate-every",
        type=int,
        default=2,
        help="Capture one animation frame every N steps. Use 0 to disable frame-sequence export.",
    )
    parser.add_argument(
        "--animation-view",
        type=str,
        default="density_xz",
        choices=["volume_front", "volume_top", "mip_top", "density_xy", "density_xz", "density_yz", "temperature_xy", "velocity_xy"],
        help="Which exported view to use for the animation sequence.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=12,
        help="Frame rate for GIF/MP4 export.",
    )
    parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Assemble captured frames into a GIF after simulation.",
    )
    parser.add_argument(
        "--make-mp4",
        action="store_true",
        help="Assemble captured frames into an MP4 with ffmpeg after simulation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of untimed warmup steps before the measured simulation loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(dt=args.dt)
    sim = StableFluids3D(cfg)
    steps = args.steps if args.steps is not None else max(1, int(round(args.sim_seconds / args.dt)))
    simulated_seconds = steps * args.dt

    print(f"triton version: {triton.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"grid: {cfg.nx}x{cfg.ny}x{cfg.nz}, device: {cfg.device}")
    print("scene: bottom spherical smoke plume, closed boundaries")
    print(f"dt={args.dt:.4f}s, steps={steps}, simulated_time={simulated_seconds:.2f}s")
    frame_dir = args.output_dir / "frames" / args.animation_view

    for _ in range(max(args.warmup, 0)):
        sim.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for step in range(1, steps + 1):
        sim.step()
        if step % args.stats_every == 0 or step == 1 or step == steps:
            torch.cuda.synchronize()
            print(f"t={step * args.dt:6.2f}s " + format_stats(step, sim))
        if args.export_every > 0 and step % args.export_every == 0:
            save_visualizations(sim, args.output_dir, step)
        if args.animate_every > 0 and step % args.animate_every == 0:
            export_animation_frame(sim, frame_dir, step, args.animation_view)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"elapsed={elapsed:.3f}s, steps_per_second={steps / elapsed:.3f}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sim.snapshot(), args.save)
        print(f"saved final state to {args.save}")
    if args.export_final:
        save_visualizations(sim, args.output_dir, steps)
        print(f"saved visualizations to {args.output_dir}")
    if args.make_gif:
        gif_path = args.output_dir / f"{args.animation_view}.gif"
        if build_gif_from_frames(frame_dir, gif_path, args.animation_fps):
            print(f"saved gif to {gif_path}")
        else:
            print(f"skipped gif export: no frames in {frame_dir}")
    if args.make_mp4:
        mp4_path = args.output_dir / f"{args.animation_view}.mp4"
        ok, reason = build_mp4_from_frames(frame_dir, mp4_path, args.animation_fps)
        if ok:
            print(f"saved mp4 to {mp4_path}")
        else:
            print(f"skipped mp4 export: {reason}")


if __name__ == "__main__":
    main()
