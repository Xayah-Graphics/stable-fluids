
module vk.camera;

import vk.math;
import std;

namespace vk::camera {

    namespace detail {

        // -------------------------
        // Small scalar helpers
        // -------------------------

        constexpr float pitch_limit_rad = 1.55334303427f;

        [[nodiscard]] inline float clampf(float x, float a, float b) noexcept {
            if (x < a) return a;
            if (x > b) return b;
            return x;
        }

        [[nodiscard]] inline float safe_aspect(std::uint32_t w, std::uint32_t h) noexcept {
            const auto fw = static_cast<float>(w ? w : 1);
            const auto fh = static_cast<float>(h ? h : 1);
            return fw / fh;
        }

        // -------------------------
        // vec3 helpers (vk.math is assumed minimal)
        // -------------------------

        [[nodiscard]] inline math::vec3 add3(math::vec3 a, math::vec3 b) noexcept {
            return vk::math::add(a, b);
        }
        [[nodiscard]] inline math::vec3 sub3(math::vec3 a, math::vec3 b) noexcept {
            return vk::math::sub(a, b);
        }
        [[nodiscard]] inline math::vec3 mul3(math::vec3 v, float s) noexcept {
            return vk::math::mul(v, s);
        }

        [[nodiscard]] inline math::vec3 normalized_or(math::vec3 v, math::vec3 fallback) noexcept {
            const float len = vk::math::length(v);
            if (len > 1e-8f) return vk::math::mul(v, 1.0f / len);
            return fallback;
        }

        // Rodrigues rotation around a unit axis.
        [[nodiscard]] inline math::vec3 rotate_axis_angle(math::vec3 v, math::vec3 axis_unit, float rad) noexcept {
            const float c = std::cos(rad);
            const float s = std::sin(rad);

            const math::vec3 term1 = mul3(v, c);
            const math::vec3 term2 = mul3(vk::math::cross(axis_unit, v), s);
            const math::vec3 term3 = mul3(axis_unit, vk::math::dot(axis_unit, v) * (1.0f - c));

            return add3(add3(term1, term2), term3);
        }

        // -------------------------
        // Convention helpers
        // -------------------------

        [[nodiscard]] inline math::vec3 axis_dir_to_vec3(AxisDir d) noexcept {
            const float s = (d.sign == Sign::Positive) ? 1.0f : -1.0f;
            switch (d.axis) {
            case Axis::X: return {s, 0.0f, 0.0f, 0.0f};
            case Axis::Y: return {0.0f, s, 0.0f, 0.0f};
            case Axis::Z: return {0.0f, 0.0f, s, 0.0f};
            }
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }

        struct Basis {
            // World-space basis corresponding to view-space axes X/Y/Z after Convention mapping.
            math::vec3 x;
            math::vec3 y;
            math::vec3 z;
        };

        // Build a stable, orthonormal frame from a desired forward direction and a world-up hint.
        // This is geometric (world) and does NOT yet apply "which view axis is forward".
        struct Frame {
            math::vec3 right;
            math::vec3 up;
            math::vec3 forward;
        };

        [[nodiscard]] inline Frame make_geometric_frame(const Convention& conv, math::vec3 forward_world) noexcept {
            forward_world = normalized_or(forward_world, {0.0f, 0.0f, -1.0f, 0.0f});

            math::vec3 up_hint = axis_dir_to_vec3(conv.world_up);
            up_hint            = normalized_or(up_hint, {0.0f, 1.0f, 0.0f, 0.0f});

            math::vec3 right;
            math::vec3 up;

            if (conv.handedness == Handedness::Right) {
                right = vk::math::cross(forward_world, up_hint);
                right = normalized_or(right, {1.0f, 0.0f, 0.0f, 0.0f});
                up    = vk::math::cross(right, forward_world);
            } else {
                right = vk::math::cross(up_hint, forward_world);
                right = normalized_or(right, {1.0f, 0.0f, 0.0f, 0.0f});
                up    = vk::math::cross(forward_world, right);
            }

            up = normalized_or(up, up_hint);
            return {right, up, forward_world};
        }

        // Map the geometric frame into the view-space axis assignment defined by Convention.
        //
        // Idea:
        //   - We already have a world-space "forward direction" (where the camera looks).
        //   - Convention says which view axis represents "forward" (X/Y/Z) and its sign.
        //   - We output world-space vectors bx/by/bz which correspond to view X/Y/Z axes.
        //
        // The math is intentionally explicit, because implicit assumptions here are a
        // common source of "my camera is rotated 90 degrees" bugs.
        [[nodiscard]] inline Basis map_frame_to_view_axes(const Convention& conv, const Frame& f) noexcept {
            math::vec3 bx;
            math::vec3 by = f.up;
            math::vec3 bz;

            const float fwd_sign = (conv.view_forward.sign == Sign::Positive) ? 1.0f : -1.0f;
            const math::vec3 fwd = mul3(f.forward, fwd_sign);

            if (conv.view_forward.axis == Axis::Z) {
                bz = normalized_or(fwd, {0.0f, 0.0f, -1.0f, 0.0f});

                if (conv.handedness == Handedness::Right) {
                    bx = normalized_or(vk::math::cross(by, bz), {1.0f, 0.0f, 0.0f, 0.0f});
                    by = normalized_or(vk::math::cross(bz, bx), by);
                } else {
                    bx = normalized_or(vk::math::cross(bz, by), {1.0f, 0.0f, 0.0f, 0.0f});
                    by = normalized_or(vk::math::cross(bx, bz), by);
                }

                return {bx, by, bz};
            }

            if (conv.view_forward.axis == Axis::X) {
                bx = normalized_or(fwd, {1.0f, 0.0f, 0.0f, 0.0f});

                if (conv.handedness == Handedness::Right) {
                    bz = normalized_or(vk::math::cross(bx, by), {0.0f, 0.0f, 1.0f, 0.0f});
                    by = normalized_or(vk::math::cross(bz, bx), by);
                } else {
                    bz = normalized_or(vk::math::cross(by, bx), {0.0f, 0.0f, 1.0f, 0.0f});
                    by = normalized_or(vk::math::cross(bx, bz), by);
                }

                return {bx, by, bz};
            }

            // view_forward.axis == Axis::Y:
            // In this special case, "forward" is the Y axis, so we treat up as forward-adjacent
            // and reconstruct the remaining axes from the geometric frame.
            bz = normalized_or(vk::math::cross(f.right, f.up), {0.0f, 0.0f, 1.0f, 0.0f});

            if (conv.handedness == Handedness::Right) {
                bx = normalized_or(vk::math::cross(by, bz), {1.0f, 0.0f, 0.0f, 0.0f});
            } else {
                bx = normalized_or(vk::math::cross(bz, by), {1.0f, 0.0f, 0.0f, 0.0f});
            }

            return {bx, by, bz};
        }

        [[nodiscard]] inline math::mat4 make_w2c(math::vec3 eye, math::vec3 bx, math::vec3 by, math::vec3 bz) noexcept {
            math::mat4 m{};
            m.c0 = {bx.x, by.x, bz.x, 0.0f};
            m.c1 = {bx.y, by.y, bz.y, 0.0f};
            m.c2 = {bx.z, by.z, bz.z, 0.0f};

            const float tx = -vk::math::dot(bx, eye);
            const float ty = -vk::math::dot(by, eye);
            const float tz = -vk::math::dot(bz, eye);

            m.c3 = {tx, ty, tz, 1.0f};
            return m;
        }

        [[nodiscard]] inline math::mat4 make_c2w(math::vec3 eye, math::vec3 bx, math::vec3 by, math::vec3 bz) noexcept {
            math::mat4 m{};
            m.c0 = {bx.x, bx.y, bx.z, 0.0f};
            m.c1 = {by.x, by.y, by.z, 0.0f};
            m.c2 = {bz.x, bz.y, bz.z, 0.0f};
            m.c3 = {eye.x, eye.y, eye.z, 1.0f};
            return m;
        }

        // Compute a world-space "look direction" from yaw/pitch given Convention.
        //
        // Convention gives:
        //   - world_up: the axis we yaw around
        //   - view_forward: the local forward direction at yaw=pitch=0
        //
        // Steps:
        //   1) yaw rotates local forward around world_up
        //   2) pitch rotates around the current right axis
        [[nodiscard]] inline math::vec3 look_from_yaw_pitch(const Convention& conv, float yaw_rad, float pitch_rad) noexcept {
            const math::vec3 up_axis       = axis_dir_to_vec3(conv.world_up);
            const math::vec3 local_forward = axis_dir_to_vec3(conv.view_forward);

            math::vec3 look = rotate_axis_angle(local_forward, up_axis, yaw_rad);

            math::vec3 right = (conv.handedness == Handedness::Right) ? vk::math::cross(look, up_axis) : vk::math::cross(up_axis, look);
            right            = normalized_or(right, {1.0f, 0.0f, 0.0f, 0.0f});

            look = rotate_axis_angle(look, right, pitch_rad);
            look = normalized_or(look, local_forward);

            return look;
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // Camera methods
    // -------------------------------------------------------------------------

    float Camera::clamp_pitch_(float rad) noexcept {
        return detail::clampf(rad, -detail::pitch_limit_rad, detail::pitch_limit_rad);
    }

    void Camera::set_config(const CameraConfig& cfg) noexcept {
        cfg_                = cfg;
        st_.orbit.pitch_rad = clamp_pitch_(st_.orbit.pitch_rad);
        st_.fly.pitch_rad   = clamp_pitch_(st_.fly.pitch_rad);

        rebuild_projection_(vw_, vh_);
        rebuild_matrices_();
    }

    void Camera::set_state(const CameraState& st) noexcept {
        st_                 = st;
        st_.orbit.pitch_rad = clamp_pitch_(st_.orbit.pitch_rad);
        st_.fly.pitch_rad   = clamp_pitch_(st_.fly.pitch_rad);

        rebuild_matrices_();
    }

    void Camera::set_mode(Mode m) noexcept {
        st_.mode = m;
        rebuild_matrices_();
    }

    void Camera::set_projection(Projection p) noexcept {
        cfg_.projection = p;
        rebuild_projection_(vw_, vh_);
        rebuild_matrices_();
    }

    void Camera::set_convention(const Convention& c) noexcept {
        cfg_.convention = c;
        rebuild_matrices_();
    }

    void Camera::home() noexcept {
        st_.mode            = Mode::Orbit;
        st_.orbit.target    = {0.0f, 0.0f, 0.0f, 0.0f};
        st_.orbit.distance  = 5.0f;
        st_.orbit.yaw_rad   = 0.0f;
        st_.orbit.pitch_rad = 0.0f;

        st_.fly.eye       = {0.0f, 0.0f, 5.0f, 0.0f};
        st_.fly.yaw_rad   = 0.0f;
        st_.fly.pitch_rad = 0.0f;

        rebuild_projection_(vw_, vh_);
        rebuild_matrices_();
    }

    const CameraConfig& Camera::config() const noexcept {
        return cfg_;
    }
    const CameraState& Camera::state() const noexcept {
        return st_;
    }
    const CameraMatrices& Camera::matrices() const noexcept {
        return m_;
    }

    void Camera::update(float dt_sec, std::uint32_t viewport_w, std::uint32_t viewport_h, const CameraInput& input) noexcept {
        vw_ = viewport_w ? viewport_w : 1;
        vh_ = viewport_h ? viewport_h : 1;

        rebuild_projection_(vw_, vh_);

        if (st_.mode == Mode::Orbit) {
            step_orbit_(input, vw_, vh_);
        } else {
            step_fly_(dt_sec, input);
        }

        rebuild_matrices_();
    }

    void Camera::rebuild_projection_(std::uint32_t w, std::uint32_t h) noexcept {
        const float aspect = detail::safe_aspect(w, h);

        if (cfg_.projection == Projection::Perspective) {
            // Assumes vk::math::perspective_vk produces Vulkan-style clip space projection.
            m_.proj = math::perspective_vk(cfg_.fov_y_rad, aspect, cfg_.znear, cfg_.zfar);
            return;
        }

        // Orthographic projection in Vulkan clip space.
        const float hh = cfg_.ortho_height * 0.5f;
        const float hw = hh * aspect;

        math::mat4 p{};
        p.c0    = {1.0f / hw, 0.0f, 0.0f, 0.0f};
        p.c1    = {0.0f, 1.0f / hh, 0.0f, 0.0f};
        p.c2    = {0.0f, 0.0f, 1.0f / (cfg_.zfar - cfg_.znear), 0.0f};
        p.c3    = {0.0f, 0.0f, -cfg_.znear / (cfg_.zfar - cfg_.znear), 1.0f};
        m_.proj = p;
    }

    void Camera::step_orbit_(const CameraInput& in, std::uint32_t vw, std::uint32_t vh) noexcept {
        // Current UI choice: "Houdini-style" orbit manipulation under Alt or Space.
        // This is kept as behavior-compatible with your original code.
        const bool houdini = in.alt || in.space;

        if (houdini && in.lmb) {
            st_.orbit.yaw_rad -= in.mouse_dx * cfg_.orbit_rotate_sens;
            st_.orbit.pitch_rad = clamp_pitch_(st_.orbit.pitch_rad - in.mouse_dy * cfg_.orbit_rotate_sens);
        }

        if (houdini && in.mmb) {
            const float base = (cfg_.projection == Projection::Orthographic) ? std::max(1e-4f, cfg_.ortho_height) : std::max(1e-4f, st_.orbit.distance);

            const float pan = base * cfg_.orbit_pan_sens;

            const auto sx   = static_cast<float>(vw ? vw : 1);
            const auto sy   = static_cast<float>(vh ? vh : 1);
            const float ndx = in.mouse_dx / sx;
            const float ndy = in.mouse_dy / sy;

            // We pan in the camera plane using (right, world_up). This matches typical DCC behavior:
            //   - horizontal drag moves along camera right
            //   - vertical drag moves along world up
            const math::vec3 up_axis = detail::axis_dir_to_vec3(cfg_.convention.world_up);
            const math::vec3 look    = detail::look_from_yaw_pitch(cfg_.convention, st_.orbit.yaw_rad, 0.0f);

            math::vec3 right_axis = (cfg_.convention.handedness == Handedness::Right) ? vk::math::cross(look, up_axis) : vk::math::cross(up_axis, look);
            right_axis            = detail::normalized_or(right_axis, {1.0f, 0.0f, 0.0f, 0.0f});

            st_.orbit.target = detail::sub3(st_.orbit.target, detail::mul3(right_axis, ndx * pan * 2.0f));
            st_.orbit.target = detail::add3(st_.orbit.target, detail::mul3(up_axis, ndy * pan * 2.0f));
        }

        // Zoom can be triggered by scroll, or RMB drag under the "houdini" modifier.
        if ((houdini && in.rmb) || (in.scroll != 0.0f)) {
            const float drag_zoom = (houdini && in.rmb) ? (-in.mouse_dy * 0.01f) : 0.0f;
            const float s         = in.scroll + drag_zoom;

            if (s != 0.0f) {
                const float factor = std::exp(-s * cfg_.orbit_zoom_sens);

                if (cfg_.projection == Projection::Perspective) {
                    st_.orbit.distance = detail::clampf(st_.orbit.distance * factor, 1e-4f, 1e6f);
                } else {
                    cfg_.ortho_height = detail::clampf(cfg_.ortho_height * factor, 1e-4f, 1e6f);
                }
            }
        }
    }

    void Camera::step_fly_(float dt, const CameraInput& in) noexcept {
        // Standard FPS behavior: RMB enables mouse look.
        if (in.rmb) {
            st_.fly.yaw_rad -= in.mouse_dx * cfg_.fly_look_sens;
            st_.fly.pitch_rad = clamp_pitch_(st_.fly.pitch_rad - in.mouse_dy * cfg_.fly_look_sens);
        }

        float speed = cfg_.fly_speed;
        if (in.shift) speed *= cfg_.fly_shift_mul;
        if (in.ctrl) speed *= cfg_.fly_ctrl_mul;

        const float move = speed * dt;

        const math::vec3 up_axis = detail::axis_dir_to_vec3(cfg_.convention.world_up);

        // Look direction from yaw/pitch.
        math::vec3 look = detail::look_from_yaw_pitch(cfg_.convention, st_.fly.yaw_rad, st_.fly.pitch_rad);

        // Build an orthonormal movement frame (right/up/look).
        math::vec3 right_axis = (cfg_.convention.handedness == Handedness::Right) ? vk::math::cross(look, up_axis) : vk::math::cross(up_axis, look);
        right_axis            = detail::normalized_or(right_axis, {1.0f, 0.0f, 0.0f, 0.0f});

        math::vec3 up_move = (cfg_.convention.handedness == Handedness::Right) ? vk::math::cross(right_axis, look) : vk::math::cross(look, right_axis);
        up_move            = detail::normalized_or(up_move, up_axis);

        if (in.forward) st_.fly.eye = detail::add3(st_.fly.eye, detail::mul3(look, move));
        if (in.backward) st_.fly.eye = detail::sub3(st_.fly.eye, detail::mul3(look, move));
        if (in.right) st_.fly.eye = detail::add3(st_.fly.eye, detail::mul3(right_axis, move));
        if (in.left) st_.fly.eye = detail::sub3(st_.fly.eye, detail::mul3(right_axis, move));
        if (in.up) st_.fly.eye = detail::add3(st_.fly.eye, detail::mul3(up_move, move));
        if (in.down) st_.fly.eye = detail::sub3(st_.fly.eye, detail::mul3(up_move, move));
    }

    void Camera::rebuild_matrices_() noexcept {
        // Step 1: determine eye position and desired forward direction in world space.
        math::vec3 eye;
        math::vec3 forward_world;

        if (st_.mode == Mode::Orbit) {
            forward_world = detail::look_from_yaw_pitch(cfg_.convention, st_.orbit.yaw_rad, st_.orbit.pitch_rad);
            eye           = detail::sub3(st_.orbit.target, detail::mul3(forward_world, st_.orbit.distance));
        } else {
            // Fly uses explicit eye; forward comes from yaw/pitch.
            eye           = st_.fly.eye;
            forward_world = detail::look_from_yaw_pitch(cfg_.convention, st_.fly.yaw_rad, st_.fly.pitch_rad);
        }

        // Step 2: build a stable geometric frame and then map it to view axes as per Convention.
        const detail::Frame f = detail::make_geometric_frame(cfg_.convention, forward_world);
        const detail::Basis b = detail::map_frame_to_view_axes(cfg_.convention, f);

        // Step 3: write outputs (basis vectors, matrices).
        m_.eye     = eye;
        m_.right   = f.right;
        m_.up      = f.up;
        m_.forward = forward_world;

        m_.w2c = detail::make_w2c(eye, b.x, b.y, b.z);
        m_.c2w = detail::make_c2w(eye, b.x, b.y, b.z);

        // Assumes vk::math::mul(mat4, mat4) exists.
        m_.view_proj = vk::math::mul(m_.proj, m_.w2c);
    }

} // namespace vk::camera
