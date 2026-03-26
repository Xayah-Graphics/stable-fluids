export module vk.math;
import std;


namespace vk::math {
    export struct alignas(8) vec2 {
        float x;
        float y;
    };

    export struct alignas(16) vec3 {
        float x;
        float y;
        float z;
        float _pad;
    };

    export struct alignas(16) vec4 {
        float x;
        float y;
        float z;
        float w;
    };

    export struct alignas(16) mat4 {
        vec4 c0;
        vec4 c1;
        vec4 c2;
        vec4 c3;
    };

    static_assert(std::is_standard_layout_v<vec2>);
    static_assert(std::is_trivially_copyable_v<vec2>);
    static_assert(sizeof(vec2) == 8);
    static_assert(alignof(vec2) == 8);

    static_assert(std::is_standard_layout_v<vec3>);
    static_assert(std::is_trivially_copyable_v<vec3>);
    static_assert(sizeof(vec3) == 16);
    static_assert(alignof(vec3) == 16);

    static_assert(std::is_standard_layout_v<vec4>);
    static_assert(std::is_trivially_copyable_v<vec4>);
    static_assert(sizeof(vec4) == 16);
    static_assert(alignof(vec4) == 16);

    static_assert(std::is_standard_layout_v<mat4>);
    static_assert(std::is_trivially_copyable_v<mat4>);
    static_assert(sizeof(mat4) == 64);
    static_assert(alignof(mat4) == 16);

    export [[nodiscard]] vec2 add(vec2 a, vec2 b) noexcept;
    export [[nodiscard]] vec2 sub(vec2 a, vec2 b) noexcept;
    export [[nodiscard]] vec2 mul(vec2 v, float s) noexcept;
    export [[nodiscard]] float dot(vec2 a, vec2 b) noexcept;
    export [[nodiscard]] float length2(vec2 v) noexcept;
    export [[nodiscard]] float length(vec2 v) noexcept;
    export [[nodiscard]] vec2 normalize(vec2 v) noexcept;

    export [[nodiscard]] vec3 add(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] vec3 sub(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] vec3 mul(vec3 v, float s) noexcept;
    export [[nodiscard]] float dot(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] vec3 cross(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] float length2(vec3 v) noexcept;
    export [[nodiscard]] float length(vec3 v) noexcept;
    export [[nodiscard]] vec3 normalize(vec3 v) noexcept;

    export [[nodiscard]] vec4 add(vec4 a, vec4 b) noexcept;
    export [[nodiscard]] vec4 mul(vec4 v, float s) noexcept;
    export [[nodiscard]] float dot(vec4 a, vec4 b) noexcept;

    export [[nodiscard]] mat4 identity_mat4() noexcept;
    export [[nodiscard]] vec4 mul(const mat4& m, vec4 v) noexcept;
    export [[nodiscard]] mat4 mul(const mat4& a, const mat4& b) noexcept;


    export [[nodiscard]] vec2 operator+(vec2 a, vec2 b) noexcept;
    export [[nodiscard]] vec2 operator-(vec2 a, vec2 b) noexcept;
    export [[nodiscard]] vec2 operator*(vec2 v, float s) noexcept;
    export [[nodiscard]] vec2 operator*(float s, vec2 v) noexcept;

    export [[nodiscard]] vec3 operator+(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] vec3 operator-(vec3 a, vec3 b) noexcept;
    export [[nodiscard]] vec3 operator*(vec3 v, float s) noexcept;
    export [[nodiscard]] vec3 operator*(float s, vec3 v) noexcept;

    export [[nodiscard]] vec4 operator+(vec4 a, vec4 b) noexcept;
    export [[nodiscard]] vec4 operator*(vec4 v, float s) noexcept;
    export [[nodiscard]] vec4 operator*(float s, vec4 v) noexcept;

    export [[nodiscard]] mat4 operator*(const mat4& a, const mat4& b) noexcept;
    export [[nodiscard]] vec4 operator*(const mat4& m, vec4 v) noexcept;

    export [[nodiscard]] mat4 translate(vec3 t) noexcept;
    export [[nodiscard]] mat4 rotate_y(float radians) noexcept;
    export [[nodiscard]] mat4 perspective_vk(float fovy, float aspect, float znear, float zfar) noexcept;
    export [[nodiscard]] mat4 look_at(vec3 eye, vec3 center, vec3 up) noexcept;
} // namespace vk::math
