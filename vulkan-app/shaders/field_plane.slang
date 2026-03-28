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

[[vk::binding(0, 0)]]
StructuredBuffer<float> fieldData;

static const uint PROJECTION_PERSPECTIVE = 0;
static const uint PROJECTION_ORTHOGRAPHIC = 1;
static const uint PLANE_XY = 0;
static const uint PLANE_XZ = 1;
static const uint PLANE_YZ = 2;

uint field_index(uint x, uint y, uint z)
{
    return (z * pc.params1.y + y) * pc.params1.x + x;
}

uint field_offset(uint x, uint y, uint z)
{
    return field_index(x, y, z);
}

float sample_field(float3 uvw)
{
    uvw = clamp(uvw, 0.0.xxx, 1.0.xxx);

    float3 p = uvw * float3(pc.params1.xyz - 1);
    uint3 lo = uint3(floor(p));
    uint3 hi = min(lo + 1, pc.params1.xyz - 1);
    float3 t = frac(p);

    float c000 = fieldData[field_offset(lo.x, lo.y, lo.z)];
    float c100 = fieldData[field_offset(hi.x, lo.y, lo.z)];
    float c010 = fieldData[field_offset(lo.x, hi.y, lo.z)];
    float c110 = fieldData[field_offset(hi.x, hi.y, lo.z)];
    float c001 = fieldData[field_offset(lo.x, lo.y, hi.z)];
    float c101 = fieldData[field_offset(hi.x, lo.y, hi.z)];
    float c011 = fieldData[field_offset(lo.x, hi.y, hi.z)];
    float c111 = fieldData[field_offset(hi.x, hi.y, hi.z)];

    float c00 = lerp(c000, c100, t.x);
    float c10 = lerp(c010, c110, t.x);
    float c01 = lerp(c001, c101, t.x);
    float c11 = lerp(c011, c111, t.x);
    float c0 = lerp(c00, c10, t.y);
    float c1 = lerp(c01, c11, t.y);
    return lerp(c0, c1, t.z);
}

void make_ray(float2 uv, out float3 rayOrigin, out float3 rayDir)
{
    if (pc.params2.x == PROJECTION_ORTHOGRAPHIC)
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

bool intersect_plane(float3 rayOrigin, float3 rayDir, out float3 hitPoint)
{
    hitPoint = 0.0.xxx;
    float denom = 0.0;
    float planeCoord = pc.params3.z;
    if (pc.params2.y == PLANE_XY)
    {
        float z = lerp(pc.volumeMin.z, pc.volumeMax.z, planeCoord);
        denom = rayDir.z;
        if (abs(denom) < 1.0e-6) return false;
        float t = (z - rayOrigin.z) / denom;
        if (t < 0.0) return false;
        hitPoint = rayOrigin + rayDir * t;
        return hitPoint.x >= pc.volumeMin.x && hitPoint.x <= pc.volumeMax.x && hitPoint.y >= pc.volumeMin.y && hitPoint.y <= pc.volumeMax.y;
    }
    if (pc.params2.y == PLANE_XZ)
    {
        float y = lerp(pc.volumeMin.y, pc.volumeMax.y, planeCoord);
        denom = rayDir.y;
        if (abs(denom) < 1.0e-6) return false;
        float t = (y - rayOrigin.y) / denom;
        if (t < 0.0) return false;
        hitPoint = rayOrigin + rayDir * t;
        return hitPoint.x >= pc.volumeMin.x && hitPoint.x <= pc.volumeMax.x && hitPoint.z >= pc.volumeMin.z && hitPoint.z <= pc.volumeMax.z;
    }

    float x = lerp(pc.volumeMin.x, pc.volumeMax.x, planeCoord);
    denom = rayDir.x;
    if (abs(denom) < 1.0e-6) return false;
    float t = (x - rayOrigin.x) / denom;
    if (t < 0.0) return false;
    hitPoint = rayOrigin + rayDir * t;
    return hitPoint.y >= pc.volumeMin.y && hitPoint.y <= pc.volumeMax.y && hitPoint.z >= pc.volumeMin.z && hitPoint.z <= pc.volumeMax.z;
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

    float3 worldPos;
    if (!intersect_plane(rayOrigin, rayDir, worldPos))
    {
        return 0.0.xxxx;
    }

    float3 uvw = (worldPos - pc.volumeMin.xyz) / max(pc.volumeMax.xyz - pc.volumeMin.xyz, 1.0e-5.xxx);
    float value = sample_field(uvw) * pc.params0.z;
    float valueMin = pc.params3.x;
    float valueMax = max(pc.params3.y, valueMin + 1.0e-5);
    float alpha = 0.0;
    float3 color = 0.0.xxx;
    if (valueMin < 0.0 && valueMax > 0.0)
    {
        float valueAbsMax = max(abs(valueMin), abs(valueMax));
        float magnitude = valueAbsMax > 1.0e-5 ? saturate(abs(value) / valueAbsMax) : 0.0;
        if (magnitude < 1.0e-4) return 0.0.xxxx;
        alpha = 1.0 - exp(-magnitude * pc.params0.w);
        color = value < 0.0 ? pc.colorA.rgb : pc.colorB.rgb;
    }
    else
    {
        float normalized = saturate((value - valueMin) / (valueMax - valueMin));
        if (normalized < 1.0e-4) return 0.0.xxxx;
        alpha = 1.0 - exp(-normalized * pc.params0.w);
        color = lerp(pc.colorA.rgb, pc.colorB.rgb, normalized);
    }
    return float4(color, alpha);
}
