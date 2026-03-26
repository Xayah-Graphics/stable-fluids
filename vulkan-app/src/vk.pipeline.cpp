module;
#include <vulkan/vulkan_raii.hpp>
module vk.pipeline;
import std;

std::vector<std::byte> vk::pipeline::read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("vk.pipeline: failed to open file: " + path);

    const auto end = f.tellg();
    if (end < 0) throw std::runtime_error("vk.pipeline: failed to size file: " + path);

    std::vector<std::byte> data(static_cast<std::size_t>(end));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f) throw std::runtime_error("vk.pipeline: failed to read file: " + path);

    return data;
}

vk::raii::ShaderModule vk::pipeline::load_shader_module(const raii::Device& device, std::span<const std::byte> spv) {
    if ((spv.size_bytes() % 4u) != 0u) throw std::runtime_error("vk.pipeline: SPIR-V size must be multiple of 4");

    const ShaderModuleCreateInfo ci{
        .codeSize = spv.size_bytes(),
        .pCode    = reinterpret_cast<const std::uint32_t*>(spv.data()),
    };

    return raii::ShaderModule{device, ci};
}

vk::pipeline::GraphicsPipeline vk::pipeline::create_graphics_pipeline(const raii::Device& device, const VertexInput& vin, const GraphicsPipelineDesc& desc, const raii::ShaderModule& shader_module, const char* vs_entry, const char* fs_entry) {
    GraphicsPipeline out{};

    std::vector<PushConstantRange> pcrs;
    if (desc.push_constant_bytes > 0) {
        pcrs.push_back(PushConstantRange{
            .stageFlags = desc.push_constant_stages,
            .offset     = 0,
            .size       = desc.push_constant_bytes,
        });
    }

    const PipelineLayoutCreateInfo plci{
        .setLayoutCount         = static_cast<std::uint32_t>(desc.set_layouts.size()),
        .pSetLayouts            = desc.set_layouts.empty() ? nullptr : desc.set_layouts.data(),
        .pushConstantRangeCount = static_cast<std::uint32_t>(pcrs.size()),
        .pPushConstantRanges    = pcrs.empty() ? nullptr : pcrs.data(),
    };

    out.layout = raii::PipelineLayout{device, plci};

    const std::array<PipelineShaderStageCreateInfo, 2> stages{{
        PipelineShaderStageCreateInfo{
            .stage  = ShaderStageFlagBits::eVertex,
            .module = *shader_module,
            .pName  = vs_entry,
        },
        PipelineShaderStageCreateInfo{
            .stage  = ShaderStageFlagBits::eFragment,
            .module = *shader_module,
            .pName  = fs_entry,
        },
    }};

    const bool has_vertices = !vin.attributes.empty();
    const PipelineVertexInputStateCreateInfo vi{
        .vertexBindingDescriptionCount   = has_vertices ? 1u : 0u,
        .pVertexBindingDescriptions      = has_vertices ? &vin.binding : nullptr,
        .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vin.attributes.size()),
        .pVertexAttributeDescriptions    = vin.attributes.empty() ? nullptr : vin.attributes.data(),
    };

    const PipelineInputAssemblyStateCreateInfo ia{
        .topology = desc.topology,
    };

    const PipelineViewportStateCreateInfo vp{
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    const PipelineRasterizationStateCreateInfo rs{
        .polygonMode = desc.polygon_mode,
        .cullMode    = desc.cull,
        .frontFace   = desc.front_face,
        .lineWidth   = 1.0f,
    };

    const PipelineMultisampleStateCreateInfo ms{
        .rasterizationSamples = SampleCountFlagBits::e1,
    };

    PipelineDepthStencilStateCreateInfo ds{};
    if (desc.use_depth) {
        ds = PipelineDepthStencilStateCreateInfo{
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp   = CompareOp::eLessOrEqual,
        };
    }

    const auto blend_att = detail::make_blend_attachment(desc.use_blend);
    const PipelineColorBlendStateCreateInfo cb{
        .attachmentCount = 1,
        .pAttachments    = &blend_att,
    };

    constexpr std::array dyn_states{
        DynamicState::eViewport,
        DynamicState::eScissor,
    };

    const PipelineDynamicStateCreateInfo dyn{
        .dynamicStateCount = static_cast<std::uint32_t>(dyn_states.size()),
        .pDynamicStates    = dyn_states.data(),
    };

    PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &desc.color_format,
    };

    if (desc.use_depth) {
        rendering.depthAttachmentFormat = desc.depth_format;
        if (detail::has_stencil(desc.depth_format)) {
            rendering.stencilAttachmentFormat = desc.depth_format;
        }
    }

    const GraphicsPipelineCreateInfo gpi{
        .pNext               = &rendering,
        .stageCount          = static_cast<std::uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pVertexInputState   = &vi,
        .pInputAssemblyState = &ia,
        .pViewportState      = &vp,
        .pRasterizationState = &rs,
        .pMultisampleState   = &ms,
        .pDepthStencilState  = desc.use_depth ? &ds : nullptr,
        .pColorBlendState    = &cb,
        .pDynamicState       = &dyn,
        .layout              = *out.layout,
    };

    out.pipeline = raii::Pipeline{device, nullptr, gpi};
    return out;
}
