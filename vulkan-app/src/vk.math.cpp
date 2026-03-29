module vk.math;
import std;


vk::math::vec3 vk::math::cross(const vec3 a, const vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f};
}


vk::math::mat4 vk::math::identity_mat4() noexcept {
    return {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    };
}
vk::math::vec4 vk::math::mul(const mat4& m, const vec4 v) noexcept {
    return {
        m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z + m.c3.x * v.w,
        m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z + m.c3.y * v.w,
        m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z + m.c3.z * v.w,
        m.c0.w * v.x + m.c1.w * v.y + m.c2.w * v.z + m.c3.w * v.w,
    };
}
vk::math::mat4 vk::math::mul(const mat4& a, const mat4& b) noexcept {
    return {
        mul(a, b.c0),
        mul(a, b.c1),
        mul(a, b.c2),
        mul(a, b.c3),
    };
}

vk::math::mat4 vk::math::operator*(const mat4& a, const mat4& b) noexcept {
    return mul(a, b);
}
vk::math::vec4 vk::math::operator*(const mat4& m, const vec4 v) noexcept {
    return mul(m, v);
}


vk::math::mat4 vk::math::translate(const vec3 t) noexcept {
    mat4 m = identity_mat4();
    m.c3.x = t.x;
    m.c3.y = t.y;
    m.c3.z = t.z;
    return m;
}

vk::math::mat4 vk::math::rotate_y(const float radians) noexcept {
    const float c = std::cos(radians);
    const float s = std::sin(radians);

    return {
        {c, 0.f, -s, 0.f},
        {0.f, 1.f, 0.f, 0.f},
        {s, 0.f, c, 0.f},
        {0.f, 0.f, 0.f, 1.f},
    };
}

vk::math::mat4 vk::math::perspective_vk(float fovy_rad, float aspect, float znear, float zfar) noexcept {
    if (!std::isfinite(fovy_rad) || !std::isfinite(aspect) || !std::isfinite(znear) || !std::isfinite(zfar)) return identity_mat4();
    if (!(aspect > 0.0f) || !(znear > 0.0f) || !(zfar > znear)) return identity_mat4();
    if (!(fovy_rad > 0.0f) || !(fovy_rad < std::numbers::pi_v<float>)) return identity_mat4();

    const float f = 1.0f / std::tan(fovy_rad * 0.5f);

    mat4 m{};
    m.c0 = {f / aspect, 0.0f, 0.0f, 0.0f};
    m.c1 = {0.0f, f, 0.0f, 0.0f};
    m.c2 = {0.0f, 0.0f, zfar / (znear - zfar), -1.0f};
    m.c3 = {0.0f, 0.0f, (zfar * znear) / (znear - zfar), 0.0f};
    return m;
}
vk::math::mat4 vk::math::look_at(const vec3 eye, const vec3 center, const vec3 up) noexcept {
    constexpr float epsilon = 1.0e-12f;

    vec3 f = normalize(sub(center, eye)); // forward
    if (!(length2(f) > epsilon)) f = {0.0f, 0.0f, -1.0f, 0.0f};

    vec3 up_axis = normalize(up);
    if (!(length2(up_axis) > epsilon)) up_axis = {0.0f, 1.0f, 0.0f, 0.0f};

    vec3 s = cross(f, up_axis); // right
    if (!(length2(s) > epsilon)) {
        const vec3 fallback_up = std::abs(f.y) < 0.99f ? vec3{0.0f, 1.0f, 0.0f, 0.0f} : vec3{1.0f, 0.0f, 0.0f, 0.0f};
        s = cross(f, fallback_up);
    }
    s = normalize(s);
    if (!(length2(s) > epsilon)) s = {1.0f, 0.0f, 0.0f, 0.0f};

    const vec3 u = cross(s, f); // up

    mat4 m{};
    m.c0 = {s.x, s.y, s.z, 0.0f};
    m.c1 = {u.x, u.y, u.z, 0.0f};
    m.c2 = {-f.x, -f.y, -f.z, 0.0f};
    m.c3 = {-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f};
    return m;
}
