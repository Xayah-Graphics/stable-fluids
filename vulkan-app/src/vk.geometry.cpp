module vk.geometry;

// vk::geometry::MeshP3C4 vk::geometry::make_sphere_p3c4(const float radius, std::uint32_t slices, std::uint32_t stacks, const math::vec4& color) {
//     MeshP3C4 mesh;
//
//     slices = std::max(3u, slices);
//     stacks = std::max(2u, stacks);
//
//     const std::uint32_t vertex_count = (stacks + 1) * (slices + 1);
//     const std::uint32_t index_count  = stacks * slices * 6;
//
//     mesh.vertices.reserve(vertex_count);
//     mesh.indices.reserve(index_count);
//
//     constexpr float pi = std::numbers::pi_v<float>;
//
//     for (std::uint32_t y = 0; y <= stacks; ++y) {
//         const float v     = static_cast<float>(y) / static_cast<float>(stacks);
//         const float phi   = pi * v;
//         const float sin_p = std::sin(phi);
//         const float cos_p = std::cos(phi);
//
//         for (std::uint32_t x = 0; x <= slices; ++x) {
//             const float u     = static_cast<float>(x) / static_cast<float>(slices);
//             const float theta = 2.0f * pi * u;
//
//             const float sin_t = std::sin(theta);
//             const float cos_t = std::cos(theta);
//
//             const math::vec3 pos{radius * sin_p * cos_t, radius * cos_p, radius * sin_p * sin_t};
//
//             mesh.vertices.push_back(VertexP3C4{.position = pos, .color = color});
//         }
//     }
//
//     const std::uint32_t stride = slices + 1;
//
//     for (std::uint32_t y = 0; y < stacks; ++y) {
//         for (std::uint32_t x = 0; x < slices; ++x) {
//             const std::uint32_t i0 = y * stride + x;
//             const std::uint32_t i1 = i0 + 1;
//             const std::uint32_t i2 = i0 + stride;
//             const std::uint32_t i3 = i2 + 1;
//
//             mesh.indices.push_back(i0);
//             mesh.indices.push_back(i2);
//             mesh.indices.push_back(i1);
//
//             mesh.indices.push_back(i1);
//             mesh.indices.push_back(i2);
//             mesh.indices.push_back(i3);
//         }
//     }
//
//     return mesh;
// }
//
// vk::geometry::MeshP3C4 vk::geometry::make_cube_p3c4(const float half_extent, const math::vec4& color) {
//     MeshP3C4 mesh;
//
//     const float h = half_extent;
//
//     mesh.vertices = {
//         {{-h, -h, -h}, color}, // 0
//         {{h, -h, -h}, color}, // 1
//         {{h, h, -h}, color}, // 2
//         {{-h, h, -h}, color}, // 3
//         {{-h, -h, h}, color}, // 4
//         {{h, -h, h}, color}, // 5
//         {{h, h, h}, color}, // 6
//         {{-h, h, h}, color} // 7
//     };
//
//     mesh.indices = {
//
//         0, 2, 1, 0, 3, 2, // -Z
//         4, 5, 6, 4, 6, 7, // +Z
//         0, 7, 3, 0, 4, 7, // -X
//         1, 2, 6, 1, 6, 5, // +X
//         0, 1, 5, 0, 5, 4, // -Y
//         3, 7, 6, 3, 6, 2
//
//     }; // +Y
//
//     return mesh;
// }
