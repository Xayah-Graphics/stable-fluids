struct PushConstants
{
    float4 eye;
    float4 right;
    float4 up;
    float4 forward;
    float4 volumeMin;
    float4 volumeMax;
    float4 backgroundBottom;
    float4 backgroundTop;
    float4 colorA;
    float4 colorB;
    float4 params0;
    uint4 params1;
    uint4 params2;
    float4 params3;
};

struct VSOut
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const uint PROJECTION_PERSPECTIVE = 0;

void make_ray(float2 uv, out float3 rayOrigin, out float3 rayDir)
{
    if (pc.params2.x != PROJECTION_PERSPECTIVE)
    {
        float hh = pc.params3.w * 0.5;
        float hw = hh * pc.params0.x;
        rayOrigin = pc.eye.xyz + uv.x * pc.right.xyz * hw + uv.y * pc.up.xyz * hh;
        rayDir = normalize(pc.forward.xyz);
        return;
    }

    rayOrigin = pc.eye.xyz;
    rayDir = normalize(
        pc.forward.xyz +
        uv.x * pc.right.xyz * pc.params0.x * pc.params0.y +
        uv.y * pc.up.xyz * pc.params0.y);
}

[shader("vertex")]
VSOut vs_main(uint vertexId : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };

    VSOut output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.ndc = positions[vertexId];
    return output;
}

[shader("fragment")]
float4 fs_main(VSOut input) : SV_Target
{
    float2 uv = float2(input.ndc.x, -input.ndc.y);
    float3 rayOrigin;
    float3 rayDir;
    make_ray(uv, rayOrigin, rayDir);
    float skyT = saturate(0.5 + 0.5 * rayDir.y);
    float3 background = lerp(pc.backgroundBottom.rgb, pc.backgroundTop.rgb, skyT);
    return float4(background, 1.0);
}
