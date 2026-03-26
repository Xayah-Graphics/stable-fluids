export module vk.geometry;
import vk.math;
import std;


namespace vk::geometry {
    export struct alignas(16) VertexP2C4 {
        math::vec2 position; // 8B
        math::vec2 _pad; // 8B pad to 16B
        math::vec4 color; // 16B
    };

    static_assert(std::is_standard_layout_v<VertexP2C4>);
    static_assert(std::is_trivially_copyable_v<VertexP2C4>);
    static_assert(sizeof(VertexP2C4) == 32);
    static_assert(alignof(VertexP2C4) == 16);

    export struct alignas(16) VertexP3C4 {
        math::vec3 position; // 16B
        math::vec4 color; // 16B
    };

    static_assert(std::is_standard_layout_v<VertexP3C4>);
    static_assert(std::is_trivially_copyable_v<VertexP3C4>);
    static_assert(sizeof(VertexP3C4) == 32);
    static_assert(alignof(VertexP3C4) == 16);

    export struct alignas(16) VertexP3C4T2 {
        math::vec3 position; // 16B
        math::vec4 color; // 16B
        math::vec2 uv; // 8B
        math::vec2 _pad; // 8B pad to 16B
    };

    static_assert(std::is_standard_layout_v<VertexP3C4T2>);
    static_assert(std::is_trivially_copyable_v<VertexP3C4T2>);
    static_assert(sizeof(VertexP3C4T2) == 48);
    static_assert(alignof(VertexP3C4T2) == 16);

    export struct alignas(16) Vertex {
        math::vec3 position; // 16B
        math::vec3 normal; // 16B
        math::vec2 uv; // 8B
        math::vec2 _pad0; // pad to 48B
        math::vec4 color; // 16B
    };

    static_assert(std::is_standard_layout_v<Vertex>);
    static_assert(std::is_trivially_copyable_v<Vertex>);
    static_assert(sizeof(Vertex) == 64);
    static_assert(alignof(Vertex) == 16);


    export template <typename VertexT>
    struct Mesh {
        std::vector<VertexT> vertices{};
        std::vector<std::uint32_t> indices{};
    };

    // export [[nodiscard]] MeshP3C4 make_sphere_p3c4(float radius, std::uint32_t slices, std::uint32_t stacks, const math::vec4& color);
    // export [[nodiscard]] MeshP3C4 make_cube_p3c4(float half_extent, const math::vec4& color);

    export template <typename VertexT>
    [[nodiscard]] Mesh<VertexT> make_sphere(float radius, std::uint32_t slices, std::uint32_t stacks, const math::vec4& color);

    export template <typename VertexT>
    [[nodiscard]] Mesh<VertexT> make_cube(float half_extent, const math::vec4& color);
} // namespace vk::geometry


namespace vk::geometry::detail {

    template <typename VertexT>
    VertexT make_vertex(const math::vec3& p, const math::vec4& c, const math::vec2&) {
        throw std::runtime_error("Unsupported vertex type in make_vertex");
    }

    template <>
    inline VertexP3C4 make_vertex<VertexP3C4>(const math::vec3& p, const math::vec4& c, const math::vec2&) {
        return VertexP3C4{
            .position = p,
            .color    = c,
        };
    }

    template <>
    inline VertexP3C4T2 make_vertex<VertexP3C4T2>(const math::vec3& p, const math::vec4& c, const math::vec2& uv) {
        return VertexP3C4T2{
            .position = p,
            .color    = c,
            .uv       = uv,
        };
    }

    template <>
    inline Vertex make_vertex<Vertex>(const math::vec3& p, const math::vec4& c, const math::vec2& uv) {
        return Vertex{
            .position = p,
            .normal   = math::normalize(p), // sphere default
            .uv       = uv,
            .color    = c,
        };
    }
} // namespace vk::geometry::detail
template <typename VertexT>
vk::geometry::Mesh<VertexT> vk::geometry::make_sphere(float radius, std::uint32_t slices, std::uint32_t stacks, const math::vec4& color) {
    Mesh<VertexT> mesh;

    slices = std::max(3u, slices);
    stacks = std::max(2u, stacks);

    constexpr float pi = std::numbers::pi_v<float>;

    for (std::uint32_t y = 0; y <= stacks; ++y) {
        const float v   = float(y) / float(stacks);
        const float phi = pi * v;

        const float sin_p = std::sin(phi);
        const float cos_p = std::cos(phi);

        for (std::uint32_t x = 0; x <= slices; ++x) {
            const float u     = float(x) / float(slices);
            const float theta = 2.0f * pi * u;

            const math::vec3 pos{
                radius * sin_p * std::cos(theta),
                radius * cos_p,
                radius * sin_p * std::sin(theta),
            };

            const math::vec2 uv{u, 1.0f - v};

            mesh.vertices.push_back(detail::make_vertex<VertexT>(pos, color, uv));
        }
    }

    const std::uint32_t stride = slices + 1;

    for (std::uint32_t y = 0; y < stacks; ++y) {
        for (std::uint32_t x = 0; x < slices; ++x) {
            const std::uint32_t i0 = y * stride + x;
            const std::uint32_t i1 = i0 + 1;
            const std::uint32_t i2 = i0 + stride;
            const std::uint32_t i3 = i2 + 1;

            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    }

    return mesh;
}
template <typename VertexT>
vk::geometry::Mesh<VertexT> vk::geometry::make_cube(float half_extent, const math::vec4& color) {
    Mesh<VertexT> mesh;

    const float h = half_extent;

    const math::vec3 positions[] = {
        {-h, -h, -h},
        {h, -h, -h},
        {h, h, -h},
        {-h, h, -h},
        {-h, -h, h},
        {h, -h, h},
        {h, h, h},
        {-h, h, h},
    };

    const math::vec2 uvs[] = {
        {0, 0},
        {1, 0},
        {1, 1},
        {0, 1},
        {0, 0},
        {1, 0},
        {1, 1},
        {0, 1},
    };

    for (int i = 0; i < 8; ++i) {
        mesh.vertices.push_back(detail::make_vertex<VertexT>(positions[i], color, uvs[i]));
    }

    mesh.indices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 7, 3, 0, 4, 7, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2};

    return mesh;
}
