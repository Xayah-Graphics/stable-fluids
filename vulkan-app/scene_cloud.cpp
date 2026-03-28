module;

#include "stable-fluids-3d.h"

#include <cuda_runtime.h>

module scene_cloud;

import app;
import std;

namespace scene_cloud {

    namespace {

        struct CloudFieldInfo {
            app::FieldInfo view{};
            uint32_t export_kind   = STABLE_FLUIDS_EXPORT_FIELD;
            bool use_density_field = false;
        };

        struct CloudSeed {
            float x;
            float y;
            float z;
            float rx;
            float ry;
            float rz;
            float amplitude;
        };

        constexpr std::array field_catalog_storage{
            CloudFieldInfo{
                .view =
                    {
                        .label  = "Cloud Density",
                        .preset =
                            {
                                .density_scale  = 10.08f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 0.66f,
                                .scalar_opacity = 5.7f,
                                .scalar_low_r   = 0.82f,
                                .scalar_low_g   = 0.87f,
                                .scalar_low_b   = 0.93f,
                                .scalar_high_r  = 0.99f,
                                .scalar_high_g  = 0.99f,
                                .scalar_high_b  = 0.98f,
                                .shaded_volume  = true,
                            },
                    },
                .export_kind      = STABLE_FLUIDS_EXPORT_FIELD,
                .use_density_field = true,
            },
            CloudFieldInfo{
                .view =
                    {
                        .label  = "Velocity Magnitude",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 0.65f,
                                .scalar_opacity = 2.0f,
                                .scalar_low_r   = 0.17f,
                                .scalar_low_g   = 0.31f,
                                .scalar_low_b   = 0.52f,
                                .scalar_high_r  = 0.98f,
                                .scalar_high_g  = 0.95f,
                                .scalar_high_b  = 0.76f,
                            },
                    },
                .export_kind = STABLE_FLUIDS_EXPORT_VELOCITY_MAGNITUDE,
            },
            CloudFieldInfo{
                .view =
                    {
                        .label  = "Pressure",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -0.10f,
                                .scalar_max     = 0.10f,
                                .scalar_opacity = 1.9f,
                                .scalar_low_r   = 0.11f,
                                .scalar_low_g   = 0.30f,
                                .scalar_low_b   = 0.72f,
                                .scalar_high_r  = 0.95f,
                                .scalar_high_g  = 0.66f,
                                .scalar_high_b  = 0.24f,
                            },
                    },
                .export_kind = STABLE_FLUIDS_EXPORT_PRESSURE,
            },
            CloudFieldInfo{
                .view =
                    {
                        .label  = "Divergence",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -12.0f,
                                .scalar_max     = 12.0f,
                                .scalar_opacity = 1.8f,
                                .scalar_low_r   = 0.10f,
                                .scalar_low_g   = 0.26f,
                                .scalar_low_b   = 0.65f,
                                .scalar_high_r  = 0.96f,
                                .scalar_high_g  = 0.42f,
                                .scalar_high_b  = 0.24f,
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
            .march_steps         = 128,
            .slice_position      = 0.76f,
            .show_velocity_plane = false,
            .background_bottom_r = 0.82f,
            .background_bottom_g = 0.91f,
            .background_bottom_b = 1.00f,
            .background_top_r    = 0.33f,
            .background_top_g    = 0.58f,
            .background_top_b    = 0.96f,
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
        context_               = nullptr;
        density_field_         = 0;
        force_x_device_        = nullptr;
        force_y_device_        = nullptr;
        force_z_device_        = nullptr;
        density_source_device_ = nullptr;
        force_x_host_.clear();
        force_z_host_.clear();
        wind_mask_.clear();
        shear_mask_.clear();
        curl_x_mask_.clear();
        curl_z_mask_.clear();
        pulse_mask_.clear();

        const std::array fields{
            StableFluidsFieldCreateDesc{
                .name          = "cloud_density",
                .diffusion     = 0.00002f,
                .dissipation   = 0.040f,
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

        force_x_host_.assign(cell_count, 0.0f);
        force_z_host_.assign(cell_count, 0.0f);
        wind_mask_.assign(cell_count, 0.0f);
        shear_mask_.assign(cell_count, 0.0f);
        curl_x_mask_.assign(cell_count, 0.0f);
        curl_z_mask_.assign(cell_count, 0.0f);
        pulse_mask_.assign(cell_count, 0.0f);

        std::vector<float> force_y_host(cell_count, 0.0f);
        std::vector<float> density_source_host(cell_count, 0.0f);
        constexpr std::array cloud_seeds{
            CloudSeed{0.09f, 0.77f, 0.18f, 0.10f, 0.045f, 0.10f, 0.82f},
            CloudSeed{0.18f, 0.79f, 0.24f, 0.08f, 0.040f, 0.08f, 0.58f},
            CloudSeed{0.31f, 0.75f, 0.58f, 0.12f, 0.050f, 0.12f, 0.88f},
            CloudSeed{0.40f, 0.78f, 0.67f, 0.09f, 0.040f, 0.09f, 0.56f},
            CloudSeed{0.53f, 0.72f, 0.34f, 0.13f, 0.055f, 0.11f, 0.84f},
            CloudSeed{0.63f, 0.74f, 0.42f, 0.08f, 0.040f, 0.07f, 0.50f},
            CloudSeed{0.74f, 0.80f, 0.76f, 0.12f, 0.050f, 0.11f, 0.90f},
            CloudSeed{0.83f, 0.77f, 0.67f, 0.08f, 0.040f, 0.08f, 0.54f},
            CloudSeed{0.91f, 0.75f, 0.27f, 0.10f, 0.045f, 0.09f, 0.76f},
        };

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto index = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                    const float px   = (static_cast<float>(x) + 0.5f) * h;
                    const float py   = (static_cast<float>(y) + 0.5f) * h;
                    const float pz   = (static_cast<float>(z) + 0.5f) * h;
                    const float fx   = px / extent_x;
                    const float fy   = py / extent_y;
                    const float fz   = pz / extent_z;

                    float layer = 0.0f;
                    const float layer_center = 0.76f;
                    const float layer_half = 0.11f;
                    const float layer_y = std::abs(fy - layer_center);
                    if (layer_y < layer_half) layer = 1.0f - layer_y / layer_half;

                    float cover = 0.0f;
                    for (const auto& seed : cloud_seeds) {
                        float dx = fx - seed.x;
                        float dz = fz - seed.z;
                        if (dx > 0.5f) dx -= 1.0f;
                        if (dx < -0.5f) dx += 1.0f;
                        if (dz > 0.5f) dz -= 1.0f;
                        if (dz < -0.5f) dz += 1.0f;
                        const float dy = fy - seed.y;
                        const float norm =
                            (dx * dx) / (seed.rx * seed.rx) +
                            (dy * dy) / (seed.ry * seed.ry) +
                            (dz * dz) / (seed.rz * seed.rz);
                        if (norm >= 1.0f) continue;
                        cover += seed.amplitude * (1.0f - norm);
                    }
                    cover = std::clamp(cover, 0.0f, 1.35f);
                    const float band = layer * cover;
                    const float wave = 0.5f + 0.5f * std::sin(fx * 11.0f + fz * 7.5f);
                    const float warp = 0.5f + 0.5f * std::cos(fx * 6.0f - fz * 9.0f);
                    const float puff = std::clamp(band * band * (1.22f + 0.28f * wave + 0.20f * warp), 0.0f, 1.38f);
                    const float dx_center = fx - 0.5f;
                    const float dz_center = fz - 0.5f;
                    const float radial = std::sqrt(dx_center * dx_center + dz_center * dz_center);
                    const float inv_radial = radial > 1.0e-5f ? 1.0f / radial : 0.0f;
                    wind_mask_[index] = puff * (0.72f + 0.28f * wave);
                    shear_mask_[index] = puff * std::sin((fy - 0.68f) * 17.0f);
                    curl_x_mask_[index] = -dz_center * inv_radial * puff * (0.35f + 0.65f * warp);
                    curl_z_mask_[index] = dx_center * inv_radial * puff * (0.35f + 0.65f * wave);
                    pulse_mask_[index] = puff * (0.65f + 0.35f * std::sin(fx * 13.0f + fz * 4.0f));
                    density_source_host[index] = 1.95f * puff;
                    force_y_host[index] = 0.055f * puff;
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
            const float phase = static_cast<float>(animation_step_) * 0.018f;
            const float wind = 0.14f + 0.02f * std::sin(phase * 0.37f);
            const float shear = 0.025f * std::sin(phase * 0.63f);
            const float drift = 0.035f * std::cos(phase * 0.29f);
            const float curl = 0.055f * std::sin(phase * 0.51f);
            for (size_t i = 0; i < force_x_host_.size(); ++i) {
                force_x_host_[i] = wind * wind_mask_[i] + shear * shear_mask_[i] + curl * curl_x_mask_[i];
                force_z_host_[i] = drift * pulse_mask_[i] + curl * curl_z_mask_[i];
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
        const auto& field = field_catalog_storage[(std::min)(static_cast<size_t>(field_index), field_catalog_storage.size() - 1)];
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

} // namespace scene_cloud
