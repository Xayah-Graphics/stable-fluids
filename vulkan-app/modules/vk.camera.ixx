export module vk.camera;

import vk.math;
import std;

namespace vk::camera {

    // -------------------------------------------------------------------------
    // High-level behavior
    // -------------------------------------------------------------------------
    //
    // This module provides a camera controller and the matrices required by a
    // renderer:
    //   - w2c (view matrix), c2w (inverse view)
    //   - proj (Vulkan-style projection), view_proj
    //   - eye/right/up/forward vectors in world space
    //
    // It supports two common interaction modes:
    //   - Orbit: camera orbits around a target point with yaw/pitch + distance
    //   - Fly  : free-fly (FPS-style) with yaw/pitch + positional movement
    //
    // A key feature is "Convention": you can define which axis is "world up",
    // which axis is "view forward", and the handedness. This is useful when you
    // want the same camera code to work across datasets/tools (Blender, COLMAP,
    // DCCs, etc.) while still outputting a consistent engine-space camera.
    // -------------------------------------------------------------------------

    export enum class Mode : std::uint8_t { Orbit, Fly };
    export enum class Projection : std::uint8_t { Perspective, Orthographic };

    export enum class Axis : std::uint8_t { X, Y, Z };
    export enum class Sign : std::int8_t { Positive = +1, Negative = -1 };
    export enum class Handedness : std::uint8_t { Right, Left };

    export struct AxisDir {
        Axis axis{};
        Sign sign{Sign::Positive};
    };

    export struct Convention {
        Handedness handedness{Handedness::Right};

        // World "up" direction. Example:
        //   - Y+ for many engines
        //   - Z+ for Blender
        AxisDir world_up{Axis::Y, Sign::Positive};

        // The camera's "forward" axis in view space, expressed as an axis+sign.
        // Many engines use -Z forward (OpenGL-style), some use +Z.
        AxisDir view_forward{Axis::Z, Sign::Negative};
    };

    export struct CameraConfig {
        Projection projection = Projection::Perspective;
        Convention convention{};

        // Perspective projection parameters (Vulkan clip space).
        float fov_y_rad = std::numbers::pi_v<float> / 3.0f;
        float znear     = 0.01f;
        float zfar      = 1000.0f;

        // Orthographic size control: height of the view volume at the camera.
        float ortho_height = 5.0f;

        // Orbit tuning.
        float orbit_rotate_sens = 0.0065f;
        float orbit_pan_sens    = 1.0f;
        float orbit_zoom_sens   = 0.12f;

        // Fly tuning.
        float fly_look_sens = 0.0045f;
        float fly_speed     = 2.0f;
        float fly_shift_mul = 3.5f;
        float fly_ctrl_mul  = 0.25f;
    };

    export struct OrbitState {
        math::vec3 target{0.0f, 0.0f, 0.0f, 0.0f};
        float distance  = 5.0f;
        float yaw_rad   = 0.0f;
        float pitch_rad = 0.0f;
    };

    export struct FlyState {
        math::vec3 eye{0.0f, 0.0f, 5.0f, 0.0f};
        float yaw_rad   = 0.0f;
        float pitch_rad = 0.0f;
    };

    export struct CameraState {
        Mode mode = Mode::Orbit;
        OrbitState orbit{};
        FlyState fly{};
    };

    export struct CameraInput {
        // Mouse buttons.
        bool lmb = false;
        bool mmb = false;
        bool rmb = false;

        // Per-frame deltas (already accumulated by the app).
        float mouse_dx = 0.0f;
        float mouse_dy = 0.0f;
        float scroll   = 0.0f;

        // Movement intents.
        bool forward  = false;
        bool backward = false;
        bool left     = false;
        bool right    = false;
        bool up       = false;
        bool down     = false;

        // Modifiers (app-defined meaning).
        bool shift = false;
        bool ctrl  = false;
        bool alt   = false;
        bool space = false;
    };

    export struct CameraMatrices {
        math::mat4 w2c{};
        math::mat4 c2w{};

        math::mat4 proj{};
        math::mat4 view_proj{};

        math::vec3 eye{0.0f, 0.0f, 0.0f, 0.0f};

        // World-space camera basis vectors.
        // right/up/forward are the geometric camera axes in world space,
        // where forward is the actual look direction from eye to target.
        math::vec3 right{1.0f, 0.0f, 0.0f, 0.0f};
        math::vec3 up{0.0f, 1.0f, 0.0f, 0.0f};
        math::vec3 forward{0.0f, 0.0f, -1.0f, 0.0f};
    };

    export class Camera {
    public:
        Camera() {}

        void set_config(const CameraConfig& cfg) noexcept;
        void set_state(const CameraState& st) noexcept;

        void set_mode(Mode m) noexcept;
        void set_projection(Projection p) noexcept;
        void set_convention(const Convention& c) noexcept;

        // Reset to a reasonable default pose for both Orbit and Fly.
        void home() noexcept;

        // Update controller state, then rebuild matrices.
        void update(float dt_sec, std::uint32_t viewport_w, std::uint32_t viewport_h, const CameraInput& input) noexcept;

        [[nodiscard]] const CameraConfig& config() const noexcept;
        [[nodiscard]] const CameraState& state() const noexcept;
        [[nodiscard]] const CameraMatrices& matrices() const noexcept;

    private:
        // Controller steps: mutate st_ only.
        void step_orbit_(const CameraInput& in, std::uint32_t vw, std::uint32_t vh) noexcept;
        void step_fly_(float dt, const CameraInput& in) noexcept;

        // Rebuild derived data: projection + view matrices + basis vectors.
        void rebuild_projection_(std::uint32_t w, std::uint32_t h) noexcept;
        void rebuild_matrices_() noexcept;

        static float clamp_pitch_(float rad) noexcept;

        CameraConfig cfg_{};
        CameraState st_{};
        CameraMatrices m_{};

        std::uint32_t vw_ = 1;
        std::uint32_t vh_ = 1;
    };

} // namespace vk::camera
