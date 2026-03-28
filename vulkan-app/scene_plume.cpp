module;

#include "stable-fluids-3d.h"

#include <cuda_runtime.h>

module scene_plume;

import app;
import std;

namespace scene_plume {

    namespace {

        struct PlumeFieldInfo {
            app::FieldInfo view{};
            uint32_t export_kind   = STABLE_FLUIDS_EXPORT_FIELD;
            bool use_density_field = false;
        };

        constexpr std::array field_catalog_storage{
            PlumeFieldInfo{
                .view =
                    {
                        .label  = "Density",
                        .preset =
                            {
                                .density_scale  = 1.35f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 3.5f,
                                .scalar_opacity = 5.4f,
                                .scalar_low_r   = 0.03f,
                                .scalar_low_g   = 0.04f,
                                .scalar_low_b   = 0.07f,
                                .scalar_high_r  = 0.94f,
                                .scalar_high_g  = 0.90f,
                                .scalar_high_b  = 0.84f,
                            },
                    },
                .export_kind      = STABLE_FLUIDS_EXPORT_FIELD,
                .use_density_field = true,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label  = "Velocity Magnitude",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 1.3f,
                                .scalar_opacity = 2.2f,
                                .scalar_low_r   = 0.06f,
                                .scalar_low_g   = 0.10f,
                                .scalar_low_b   = 0.24f,
                                .scalar_high_r  = 0.24f,
                                .scalar_high_g  = 0.88f,
                                .scalar_high_b  = 1.00f,
                            },
                    },
                .export_kind = STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label  = "Pressure",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -0.18f,
                                .scalar_max     = 0.18f,
                                .scalar_opacity = 2.3f,
                                .scalar_low_r   = 0.08f,
                                .scalar_low_g   = 0.22f,
                                .scalar_low_b   = 0.62f,
                                .scalar_high_r  = 0.96f,
                                .scalar_high_g  = 0.58f,
                                .scalar_high_b  = 0.18f,
                            },
                    },
                .export_kind = STABLE_FLUIDS_EXPORT_PRESSURE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label  = "Divergence",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -24.0f,
                                .scalar_max     = 24.0f,
                                .scalar_opacity = 2.3f,
                                .scalar_low_r   = 0.05f,
                                .scalar_low_g   = 0.14f,
                                .scalar_low_b   = 0.50f,
                                .scalar_high_r  = 0.94f,
                                .scalar_high_g  = 0.28f,
                                .scalar_high_b  = 0.22f,
                            },
                    },
                .export_kind = STABLE_FLUIDS_EXPORT_DIVERGENCE,
            },
        };
        constexpr auto field_views = [] {
            std::array<app::FieldInfo, field_catalog_storage.size()> result{};
            for (size_t i = 0; i < result.size(); ++i) result[i] = field_catalog_storage[i].view;
            return result;
        }();

    } // namespace

    Scene::Scene() {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    Scene::~Scene() {
        if (context_ != nullptr) stable_fluids_destroy_context_cuda(context_);
        if (force_x_device_ != nullptr) cudaFree(force_x_device_);
        if (force_y_device_ != nullptr) cudaFree(force_y_device_);
        if (force_z_device_ != nullptr) cudaFree(force_z_device_);
        if (density_source_device_ != nullptr) cudaFree(density_source_device_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    std::span<const app::FieldInfo> Scene::fields() const {
        return std::span<const app::FieldInfo>{field_views};
    }

    app::VisualizationSettings Scene::default_visualization() const {
        app::VisualizationSettings settings{
            .view_mode           = app::ViewMode::Volume,
            .plane_axis          = app::PlaneAxis::XY,
            .march_steps         = 112,
            .slice_position      = 0.42f,
            .show_velocity_plane = false,
            .background_bottom_r = 0.0f,
            .background_bottom_g = 0.0f,
            .background_bottom_b = 0.0f,
            .background_top_r    = 0.0f,
            .background_top_g    = 0.0f,
            .background_top_b    = 0.0f,
        };
        app::apply_field_preset(settings, field_catalog_storage[0].view.preset);
        return settings;
    }

    app::SceneInfo Scene::info() const {
        return info_;
    }

    cudaStream_t Scene::stream() const {
        return stream_;
    }

    void Scene::rebuild() {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        if (context_ != nullptr) check_stable(stable_fluids_destroy_context_cuda(context_), "stable_fluids_destroy_context_cuda");
        if (force_x_device_ != nullptr) cudaFree(force_x_device_);
        if (force_y_device_ != nullptr) cudaFree(force_y_device_);
        if (force_z_device_ != nullptr) cudaFree(force_z_device_);
        if (density_source_device_ != nullptr) cudaFree(density_source_device_);
        context_                = nullptr;
        density_field_          = 0;
        force_x_device_         = nullptr;
        force_y_device_         = nullptr;
        force_z_device_         = nullptr;
        density_source_device_  = nullptr;
        force_x_host_.clear();
        force_z_host_.clear();
        source_mask_.clear();
        swirl_x_mask_.clear();
        swirl_z_mask_.clear();
        drift_mask_.clear();

        const std::array fields{
            StableFluidsFieldCreateDesc{
                .name          = "density",
                .diffusion     = 0.00005f,
                .dissipation   = 0.35f,
                .initial_value = 0.0f,
            },
        };
        std::array<StableFluidsFieldHandle, 1> field_handles{};
        const StableFluidsContextCreateDesc create_desc{
            .config      = config_,
            .stream      = stream_,
            .fields      = fields.data(),
            .field_count = static_cast<uint32_t>(fields.size()),
        };
        check_stable(stable_fluids_create_context_cuda(&create_desc, &context_, field_handles.data(), static_cast<uint32_t>(field_handles.size())), "stable_fluids_create_context_cuda");
        density_field_ = field_handles[0];

        const auto nx           = config_.nx;
        const auto ny           = config_.ny;
        const auto nz           = config_.nz;
        const auto cell_count   = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
        const auto scalar_bytes = cell_count * sizeof(float);
        const float h           = config_.cell_size;
        const float extent_x    = static_cast<float>(nx) * h;
        const float extent_y    = static_cast<float>(ny) * h;
        const float extent_z    = static_cast<float>(nz) * h;
        grid_                   = {
                              .nx        = static_cast<uint32_t>(nx),
                              .ny        = static_cast<uint32_t>(ny),
                              .nz        = static_cast<uint32_t>(nz),
                              .cell_size = h,
        };
        const float source_x = extent_x * 0.50f;
        const float source_y = extent_y * 0.13f;
        const float source_z = extent_z * 0.50f;
        const float source_r = h * 6.0f;
        const float swirl_y  = source_y + source_r * 0.90f;
        const float drift_y  = source_y + source_r * 1.35f;

        force_x_host_.assign(cell_count, 0.0f);
        force_z_host_.assign(cell_count, 0.0f);
        source_mask_.assign(cell_count, 0.0f);
        swirl_x_mask_.assign(cell_count, 0.0f);
        swirl_z_mask_.assign(cell_count, 0.0f);
        drift_mask_.assign(cell_count, 0.0f);

        std::vector<float> force_y_host(cell_count, 0.0f);
        std::vector<float> density_source_host(cell_count, 0.0f);
        auto radial_weight = [](const float px, const float py, const float pz, const float cx, const float cy, const float cz, const float radius) {
            const float dx      = px - cx;
            const float dy      = py - cy;
            const float dz      = pz - cz;
            const float radius2 = radius * radius;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            if (dist2 >= radius2 || radius2 <= 0.0f) return 0.0f;
            return 1.0f - dist2 / radius2;
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto index          = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                    const float px            = (static_cast<float>(x) + 0.5f) * h;
                    const float py            = (static_cast<float>(y) + 0.5f) * h;
                    const float pz            = (static_cast<float>(z) + 0.5f) * h;
                    const float source_weight = radial_weight(px, py, pz, source_x, source_y, source_z, source_r);
                    const float swirl_weight  = radial_weight(px, py, pz, source_x, swirl_y, source_z, source_r * 1.65f);
                    const float drift_weight  = radial_weight(px, py, pz, source_x, drift_y, source_z, source_r * 2.10f);
                    const float dx            = px - source_x;
                    const float dz            = pz - source_z;
                    const float radial        = std::sqrt(dx * dx + dz * dz);
                    const float inv_radial    = radial > 1.0e-5f ? 1.0f / radial : 0.0f;
                    source_mask_[index]       = source_weight;
                    swirl_x_mask_[index]      = -dz * inv_radial * swirl_weight;
                    swirl_z_mask_[index]      = dx * inv_radial * swirl_weight;
                    drift_mask_[index]        = drift_weight;
                    density_source_host[index] = 32.0f * source_weight;
                    force_y_host[index]        = 7.6f * source_weight;
                }
            }
        }

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_x_device_), scalar_bytes), "cudaMalloc force_x_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_y_device_), scalar_bytes), "cudaMalloc force_y_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_z_device_), scalar_bytes), "cudaMalloc force_z_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&density_source_device_), scalar_bytes), "cudaMalloc density_source_device");
        check_cuda(cudaMemsetAsync(force_x_device_, 0, scalar_bytes, stream_), "cudaMemsetAsync force_x_device");
        check_cuda(cudaMemsetAsync(force_z_device_, 0, scalar_bytes, stream_), "cudaMemsetAsync force_z_device");
        check_cuda(cudaMemcpyAsync(force_y_device_, force_y_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_y_device");
        check_cuda(cudaMemcpyAsync(density_source_device_, density_source_host.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync density_source_device");
        animation_step_ = 0;
        info_ = {
            .grid              = grid_,
            .dt                = config_.dt,
            .step_count        = 0,
            .last_step_call_ms = 0.0,
        };
    }

    void Scene::step(const int sim_steps) {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        if (sim_steps <= 0) return;
        const auto scalar_bytes = force_x_host_.size() * sizeof(float);
        const StableFluidsFieldSourceDesc field_source{
            .field  = density_field_,
            .values = density_source_device_,
        };

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            const float phase   = static_cast<float>(animation_step_) * 0.045f;
            const float drift_x = 1.15f * std::sin(phase);
            const float drift_z = 0.85f * std::cos(phase * 0.71f);
            const float swirl   = 1.65f * std::cos(phase * 0.53f);
            for (size_t i = 0; i < force_x_host_.size(); ++i) {
                force_x_host_[i] = drift_x * drift_mask_[i] + swirl * swirl_x_mask_[i];
                force_z_host_[i] = drift_z * drift_mask_[i] + swirl * swirl_z_mask_[i];
            }

            const auto begin = std::chrono::steady_clock::now();
            check_cuda(cudaMemcpyAsync(force_x_device_, force_x_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_x_device");
            check_cuda(cudaMemcpyAsync(force_z_device_, force_z_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_z_device");
            const StableFluidsStepDesc step_desc{
                .force_x            = force_x_device_,
                .force_y            = force_y_device_,
                .force_z            = force_z_device_,
                .field_sources      = &field_source,
                .field_source_count = 1,
            };
            check_stable(stable_fluids_step_cuda(context_, &step_desc), "stable_fluids_step_cuda");
            info_.last_step_call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            ++info_.step_count;
            ++animation_step_;
        }
    }

    void Scene::export_field(const uint32_t field_index, void* const device_destination) const {
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        const auto& field = field_catalog_storage[(std::min) (static_cast<size_t>(field_index), field_catalog_storage.size() - 1)];
        const StableFluidsExportDesc export_desc{
            .kind  = field.export_kind,
            .field = field.use_density_field ? density_field_ : 0u,
        };
        check_stable(stable_fluids_export_cuda(context_, &export_desc, device_destination), "stable_fluids_export_cuda");
    }

    void Scene::export_velocity(void* const device_destination, float* const host_destination) const {
        auto check_cuda = [](const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        };
        auto check_stable = [](const StableFluidsResult code, const std::string_view what) {
            if (code == STABLE_FLUIDS_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        };
        const StableFluidsExportDesc export_desc{
            .kind = STABLE_FLUIDS_EXPORT_VELOCITY,
        };
        check_stable(stable_fluids_export_cuda(context_, &export_desc, device_destination), "stable_fluids_export_cuda");
        if (host_destination == nullptr) return;
        const auto velocity_bytes = static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz) * 3u * sizeof(float);
        check_cuda(cudaMemcpyAsync(host_destination, device_destination, velocity_bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync velocity snapshot");
    }

} // namespace scene_plume
