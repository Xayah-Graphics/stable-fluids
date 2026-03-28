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

uint field_index(uint x, uint y, uint z)
{
    return (z * pc.params1.y + y) * pc.params1.x + x;
}

float sample_field(float3 uvw)
{
    uvw = clamp(uvw, 0.0.xxx, 1.0.xxx);

    float3 p = uvw * float3(pc.params1.xyz - 1);
    uint3 lo = uint3(floor(p));
    uint3 hi = min(lo + 1, pc.params1.xyz - 1);
    float3 t = frac(p);

    float c000 = fieldData[field_index(lo.x, lo.y, lo.z)];
    float c100 = fieldData[field_index(hi.x, lo.y, lo.z)];
    float c010 = fieldData[field_index(lo.x, hi.y, lo.z)];
    float c110 = fieldData[field_index(hi.x, hi.y, lo.z)];
    float c001 = fieldData[field_index(lo.x, lo.y, hi.z)];
    float c101 = fieldData[field_index(hi.x, lo.y, hi.z)];
    float c011 = fieldData[field_index(lo.x, hi.y, hi.z)];
    float c111 = fieldData[field_index(hi.x, hi.y, hi.z)];

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
    if (t1 <= t0) return 0.0.xxxx;

    float distanceInside = t1 - t0;
    uint steps = max(pc.params1.w, 1u);
    float stepLength = distanceInside / steps;
    float transmittance = 1.0;
    float3 color = 0.0.xxx;
    float valueMin = pc.params3.x;
    float valueMax = max(pc.params3.y, valueMin + 1.0e-5);
    float3 voxel = 1.0.xxx / max(float3(pc.params1.xyz - 1), 1.0.xxx);
    const float3 lightDir = normalize(float3(-0.45, 0.82, -0.34));

    [loop]
    for (uint i = 0; i < steps; ++i)
    {
        float travel = t0 + (i + 0.5) * stepLength;
        float3 worldPos = rayOrigin + travel * rayDir;
        float3 uvw = (worldPos - pc.volumeMin.xyz) / max(pc.volumeMax.xyz - pc.volumeMin.xyz, 1.0e-5.xxx);
        float value = sample_field(uvw) * pc.params0.z;
        if (valueMin < 0.0 && valueMax > 0.0)
        {
            float valueAbsMax = max(abs(valueMin), abs(valueMax));
            float magnitude = valueAbsMax > 1.0e-5 ? saturate(abs(value) / valueAbsMax) : 0.0;
            if (magnitude < 1.0e-4) continue;
            float alpha = 1.0 - exp(-magnitude * pc.params0.w * stepLength);
            float3 sampleColor = value < 0.0 ? pc.colorA.rgb : pc.colorB.rgb;
            color += transmittance * alpha * sampleColor;
            transmittance *= 1.0 - alpha;
        }
        else
        {
            float normalized = saturate((value - valueMin) / (valueMax - valueMin));
            if (normalized < 1.0e-4) continue;
            float alpha = 1.0 - exp(-normalized * pc.params0.w * stepLength);
            float3 sampleColor = lerp(pc.colorA.rgb, pc.colorB.rgb, normalized);
            float dx = sample_field(clamp(uvw + float3(voxel.x, 0.0, 0.0), 0.0.xxx, 1.0.xxx)) * pc.params0.z - value;
            float dy = sample_field(clamp(uvw + float3(0.0, voxel.y, 0.0), 0.0.xxx, 1.0.xxx)) * pc.params0.z - value;
            float dz = sample_field(clamp(uvw + float3(0.0, 0.0, voxel.z), 0.0.xxx, 1.0.xxx)) * pc.params0.z - value;
            float3 normal = normalize(float3(-dx, -dy, -dz) + 1.0e-5.xxx);
            float diffuse = saturate(dot(normal, lightDir));
            float probe = sample_field(clamp(uvw + lightDir * 0.035, 0.0.xxx, 1.0.xxx)) * pc.params0.z;
            float shadow = exp(-probe * 1.4);
            float lighting = 0.42 + 0.48 * diffuse + 0.22 * shadow;
            float silver = pow(saturate(1.0 - dot(normal, -rayDir)), 3.0);
            sampleColor *= lighting;
            sampleColor += 0.18 * silver.xxx;
            color += transmittance * alpha * sampleColor;
            transmittance *= 1.0 - alpha;
        }

        if (transmittance < 0.01) break;
    }

    return float4(color, 1.0 - transmittance);
}
