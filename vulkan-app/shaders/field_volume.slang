struct PushConstants
{
    float4 eye;
    float4 right;
    float4 up;
    float4 forward;
    float4 volumeMin;
    float4 volumeMax;
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

static const uint RENDER_MODE_SMOKE = 0;
static const uint PROJECTION_PERSPECTIVE = 0;

uint field_index(uint x, uint y, uint z)
{
    return (z * pc.params1.y + y) * pc.params1.x + x;
}

uint field_offset(uint x, uint y, uint z)
{
    return field_index(x, y, z) * max(pc.params2.y, 1u);
}

float4 sample_field(float3 uvw)
{
    uvw = clamp(uvw, 0.0.xxx, 1.0.xxx);

    float3 p = uvw * float3(pc.params1.xyz - 1);
    uint3 lo = uint3(floor(p));
    uint3 hi = min(lo + 1, pc.params1.xyz - 1);
    float3 t = frac(p);

    float4 c000 = 0.0.xxxx;
    float4 c100 = 0.0.xxxx;
    float4 c010 = 0.0.xxxx;
    float4 c110 = 0.0.xxxx;
    float4 c001 = 0.0.xxxx;
    float4 c101 = 0.0.xxxx;
    float4 c011 = 0.0.xxxx;
    float4 c111 = 0.0.xxxx;

    [unroll]
    for (uint component = 0; component < 4; ++component)
    {
        if (component >= pc.params2.y) break;
        c000[component] = fieldData[field_offset(lo.x, lo.y, lo.z) + component];
        c100[component] = fieldData[field_offset(hi.x, lo.y, lo.z) + component];
        c010[component] = fieldData[field_offset(lo.x, hi.y, lo.z) + component];
        c110[component] = fieldData[field_offset(hi.x, hi.y, lo.z) + component];
        c001[component] = fieldData[field_offset(lo.x, lo.y, hi.z) + component];
        c101[component] = fieldData[field_offset(hi.x, lo.y, hi.z) + component];
        c011[component] = fieldData[field_offset(lo.x, hi.y, hi.z) + component];
        c111[component] = fieldData[field_offset(hi.x, hi.y, hi.z) + component];
    }

    float4 c00 = lerp(c000, c100, t.x);
    float4 c10 = lerp(c010, c110, t.x);
    float4 c01 = lerp(c001, c101, t.x);
    float4 c11 = lerp(c011, c111, t.x);
    float4 c0 = lerp(c00, c10, t.y);
    float4 c1 = lerp(c01, c11, t.y);
    return lerp(c0, c1, t.z);
}

void make_ray(float2 uv, out float3 rayOrigin, out float3 rayDir)
{
    if (pc.params2.w != PROJECTION_PERSPECTIVE)
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

float2 intersect_aabb(float3 rayOrigin, float3 rayDir, float3 boundsMin, float3 boundsMax)
{
    float3 safeDir = float3(
        abs(rayDir.x) < 1.0e-6 ? (rayDir.x < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.x,
        abs(rayDir.y) < 1.0e-6 ? (rayDir.y < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.y,
        abs(rayDir.z) < 1.0e-6 ? (rayDir.z < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.z);
    float3 invDir = 1.0.xxx / safeDir;
    float3 t0 = (boundsMin - rayOrigin) * invDir;
    float3 t1 = (boundsMax - rayOrigin) * invDir;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);

    float entry = max(max(tMin.x, tMin.y), tMin.z);
    float exit = min(min(tMax.x, tMax.y), tMax.z);
    return float2(entry, exit);
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

    float2 hit = intersect_aabb(rayOrigin, rayDir, pc.volumeMin.xyz, pc.volumeMax.xyz);
    float t0 = max(hit.x, 0.0);
    float t1 = hit.y;
    if (t1 <= t0)
    {
        return float4(0.0.xxx, 1.0);
    }

    float distanceInside = t1 - t0;
    uint steps = max(pc.params1.w, 1u);
    float stepLength = distanceInside / steps;
    float transmittance = 1.0;
    float3 color = 0.0.xxx;
    float valueMin = pc.params3.x;
    float valueMax = max(pc.params3.y, valueMin + 1.0e-5);

    [loop]
    for (uint i = 0; i < steps; ++i)
    {
        float travel = t0 + (i + 0.5) * stepLength;
        float3 worldPos = rayOrigin + travel * rayDir;
        float3 uvw = (worldPos - pc.volumeMin.xyz) / max(pc.volumeMax.xyz - pc.volumeMin.xyz, 1.0e-5.xxx);
        float4 sampleData = sample_field(uvw);

        if (pc.params2.x == RENDER_MODE_SMOKE && pc.params2.y == 4)
        {
            float density = max(sampleData.x * pc.params0.z, 0.0);
            if (density < 1.0e-4) continue;

            float3 dyeRgb = max(sampleData.yzw, 0.0.xxx);
            float3 sampleColor = saturate(dyeRgb / max(sampleData.x, 1.0e-6));
            float alpha = 1.0 - exp(-density * pc.params0.w * stepLength);
            color += transmittance * alpha * sampleColor;
            transmittance *= 1.0 - alpha;
        }
        else
        {
            float value = sampleData.x;
            if (pc.params2.y > 1 && pc.params2.y < 4) value = length(sampleData.xyz);
            value *= pc.params0.z;
            float normalized = saturate((value - valueMin) / (valueMax - valueMin));
            if (normalized < 1.0e-4) continue;
            float alpha = 1.0 - exp(-normalized * pc.params0.w * stepLength);
            float3 sampleColor = lerp(pc.colorA.rgb, pc.colorB.rgb, normalized);
            color += transmittance * alpha * sampleColor;
            transmittance *= 1.0 - alpha;
        }

        if (transmittance < 0.01) break;
    }

    return float4(color, 1.0);
}
