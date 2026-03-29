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

    export struct alignas(16) uvec4 {
        std::uint32_t x;
        std::uint32_t y;
        std::uint32_t z;
        std::uint32_t w;
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

    static_assert(std::is_standard_layout_v<uvec4>);
    static_assert(std::is_trivially_copyable_v<uvec4>);
    static_assert(sizeof(uvec4) == 16);
    static_assert(alignof(uvec4) == 16);

    static_assert(std::is_standard_layout_v<mat4>);
    static_assert(std::is_trivially_copyable_v<mat4>);
    static_assert(sizeof(mat4) == 64);
    static_assert(alignof(mat4) == 16);

    template <typename V>
    struct vec_traits;

    template <>
    struct vec_traits<vec2> {
        using scalar_type                  = float;
        static constexpr std::size_t lanes = 2;
        static scalar_type get(const vec2& v, const std::size_t i) noexcept {
            return i == 0 ? v.x : v.y;
        }
        static void set(vec2& v, const std::size_t i, const scalar_type s) noexcept {
            if (i == 0) v.x = s;
            if (i == 1) v.y = s;
        }
    };

    template <>
    struct vec_traits<vec3> {
        using scalar_type                  = float;
        static constexpr std::size_t lanes = 3;
        static scalar_type get(const vec3& v, const std::size_t i) noexcept {
            return i == 0 ? v.x : (i == 1 ? v.y : v.z);
        }
        static void set(vec3& v, const std::size_t i, const scalar_type s) noexcept {
            if (i == 0) v.x = s;
            if (i == 1) v.y = s;
            if (i == 2) v.z = s;
            v._pad = 0.0f;
        }
    };

    template <>
    struct vec_traits<vec4> {
        using scalar_type                  = float;
        static constexpr std::size_t lanes = 4;
        static scalar_type get(const vec4& v, const std::size_t i) noexcept {
            return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
        }
        static void set(vec4& v, const std::size_t i, const scalar_type s) noexcept {
            if (i == 0) v.x = s;
            if (i == 1) v.y = s;
            if (i == 2) v.z = s;
            if (i == 3) v.w = s;
        }
    };

    template <>
    struct vec_traits<uvec4> {
        using scalar_type                  = std::uint32_t;
        static constexpr std::size_t lanes = 4;
        static scalar_type get(const uvec4& v, const std::size_t i) noexcept {
            return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
        }
        static void set(uvec4& v, const std::size_t i, const scalar_type s) noexcept {
            if (i == 0) v.x = s;
            if (i == 1) v.y = s;
            if (i == 2) v.z = s;
            if (i == 3) v.w = s;
        }
    };

    template <typename V>
    concept vector_type = requires(V v, std::size_t i, typename vec_traits<V>::scalar_type s) {
        typename vec_traits<V>::scalar_type;
        { vec_traits<V>::get(v, i) } -> std::same_as<typename vec_traits<V>::scalar_type>;
        { vec_traits<V>::set(v, i, s) } -> std::same_as<void>;
    };

    template <typename V>
    concept float_vector_type = vector_type<V> && std::same_as<typename vec_traits<V>::scalar_type, float>;

    template <typename V>
    concept subtractable_vector_type = vector_type<V> && (std::floating_point<typename vec_traits<V>::scalar_type> || std::signed_integral<typename vec_traits<V>::scalar_type>);

    export template <vector_type V>
    [[nodiscard]] V add(const V a, const V b) noexcept {
        V out{};
        for (std::size_t i = 0; i < vec_traits<V>::lanes; ++i) vec_traits<V>::set(out, i, vec_traits<V>::get(a, i) + vec_traits<V>::get(b, i));
        return out;
    }

    export template <subtractable_vector_type V>
    [[nodiscard]] V sub(const V a, const V b) noexcept {
        V out{};
        for (std::size_t i = 0; i < vec_traits<V>::lanes; ++i) vec_traits<V>::set(out, i, vec_traits<V>::get(a, i) - vec_traits<V>::get(b, i));
        return out;
    }

    export template <vector_type V>
    [[nodiscard]] V mul(const V v, const typename vec_traits<V>::scalar_type s) noexcept {
        V out{};
        for (std::size_t i = 0; i < vec_traits<V>::lanes; ++i) vec_traits<V>::set(out, i, vec_traits<V>::get(v, i) * s);
        return out;
    }

    export template <vector_type V>
    [[nodiscard]] auto dot(const V a, const V b) noexcept -> vec_traits<V>::scalar_type {
        typename vec_traits<V>::scalar_type out{};
        for (std::size_t i = 0; i < vec_traits<V>::lanes; ++i) out += vec_traits<V>::get(a, i) * vec_traits<V>::get(b, i);
        return out;
    }

    export template <float_vector_type V>
    [[nodiscard]] float length2(const V v) noexcept {
        return dot(v, v);
    }

    export template <float_vector_type V>
    [[nodiscard]] float length(const V v) noexcept {
        return std::sqrt(length2(v));
    }

    export template <float_vector_type V>
    [[nodiscard]] V normalize(const V v) noexcept {
        const float l2 = length2(v);
        if (!(l2 > 0.0f)) return V{};
        return mul(v, 1.0f / std::sqrt(l2));
    }

    export [[nodiscard]] vec3 cross(vec3 a, vec3 b) noexcept;

    export [[nodiscard]] mat4 identity_mat4() noexcept;
    export [[nodiscard]] vec4 mul(const mat4& m, vec4 v) noexcept;
    export [[nodiscard]] mat4 mul(const mat4& a, const mat4& b) noexcept;


    export template <vector_type V>
    [[nodiscard]] V operator+(const V a, const V b) noexcept {
        return add(a, b);
    }

    export template <subtractable_vector_type V>
    [[nodiscard]] V operator-(const V a, const V b) noexcept {
        return sub(a, b);
    }

    export template <vector_type V>
    [[nodiscard]] V operator*(const V v, const typename vec_traits<V>::scalar_type s) noexcept {
        return mul(v, s);
    }

    export template <vector_type V>
    [[nodiscard]] V operator*(const typename vec_traits<V>::scalar_type s, const V v) noexcept {
        return mul(v, s);
    }

    export [[nodiscard]] mat4 operator*(const mat4& a, const mat4& b) noexcept;
    export [[nodiscard]] vec4 operator*(const mat4& m, vec4 v) noexcept;

    export [[nodiscard]] mat4 translate(vec3 t) noexcept;
    export [[nodiscard]] mat4 rotate_y(float radians) noexcept;
    export [[nodiscard]] mat4 perspective_vk(float fovy, float aspect, float znear, float zfar) noexcept;
    export [[nodiscard]] mat4 look_at(vec3 eye, vec3 center, vec3 up) noexcept;
} // namespace vk::math
