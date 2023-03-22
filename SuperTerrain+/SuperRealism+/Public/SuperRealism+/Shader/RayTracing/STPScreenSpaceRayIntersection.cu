#include "STPScreenSpaceRayIntersection.cuh"

#include "STPFragmentUtility.cuh"

//OptiX
#include <optix.h>

//System
#include <cuda/std/limits>

using namespace SuperTerrainPlus::STPRealism;

extern __constant__ STPScreenSpaceRayIntersectionData SSRIData;

//Ray data passed between each shader
struct STPSSRIPayload {
public:

	//The total number of payload register used
	constexpr static unsigned int DataCount = 4u;

	unsigned int PrimitiveID;
	float RayTime;
	float2 UV;

};

__device__ __forceinline__ static STPSSRIPayload traceIntersection(const float3& origin, const float3& direction, const unsigned int mask) {
	//initialise all payloads as undefined
	unsigned int p[STPSSRIPayload::DataCount];
	for (unsigned int i = 0u; i < STPSSRIPayload::DataCount; i++) {
		p[i] = optixUndefinedValue();
	}
	//unlike rasterisation, back face culling may incur performance penalty
	optixTrace(SSRIData.Handle,
		origin, direction,
		0.0f, 1e16f, 0.0f,
		mask, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0u, 0u, 0u,
		p[0], p[1], p[2], p[3]
	);

	//read payloads
	STPSSRIPayload data;
	data.PrimitiveID = p[0];
	data.RayTime = __uint_as_float(p[1]);
	data.UV = make_float2(
		__uint_as_float(p[2]),
		__uint_as_float(p[3])
	);
	return data;
}

__device__ __forceinline__ static void setPrimitiveData(const STPSSRIPayload& data) {
	//record intersection information
	optixSetPayload_0(data.PrimitiveID);
	optixSetPayload_1(__float_as_uint(data.RayTime));
	optixSetPayload_2(__float_as_uint(data.UV.x));
	optixSetPayload_3(__float_as_uint(data.UV.y));
}

__device__ __forceinline__ static void setEnvironmentData() {
	//primitiveID zero is reserved for no intersection,
	//which either denotes an environment pixel, or ray is invisible
	optixSetPayload_0(STPScreenSpaceRayIntersectionData::EnvironmentRayID);
	optixSetPayload_1(optixUndefinedValue());
	optixSetPayload_2(optixUndefinedValue());
	optixSetPayload_3(optixUndefinedValue());
}

__global__ void __raygen__launchScreenSpaceRay() {
	//we are always using 2D launch, so ignore the third component
	const uint2 idx = make_uint2(optixGetLaunchIndex()),
		dim = make_uint2(optixGetLaunchDimensions());

	const auto& textureData = SSRIData.SSTexture;
	const float2 texCoord = make_float2(idx);
	const unsigned int texIndex = texCoord.x + texCoord.y * dim.x;
	//recover fragment values from texture, using un-normalised UV
	const auto stencil = textureData.SSStencil[texIndex];
	const auto ray_depth = tex2D<float>(textureData.SSRayDepth, texCoord.x, texCoord.y);
	const auto ray_dir = make_float3(tex2D<float4>(textureData.SSRayDirection, texCoord.x, texCoord.y)) * 2.0f - 1.0f;
	//calculate ray origin using normalised UV
	const float2 uv = STPFragmentUtility::calcTextureCoordinate(idx, dim);
	const float3 ray_ori = STPFragmentUtility::reconstructDepthToWorld(SSRIData.InvProjectionView, ray_depth, uv);

	//stencil test
	const unsigned char stencil_result = STPScreenSpaceRayIntersectionData::RayVisibilityMask & stencil;
	const unsigned int rayVisibility = stencil_result == 0u ? 0x00u : 0xFFu;

	//start the magic
	//doing a branch based on stencil test on trace function gives poor performance,
	//instead we can interpret stencil as ray visibility.
	const STPSSRIPayload data = traceIntersection(ray_ori, ray_dir, rayVisibility);
	
	//store to texture
	if (!rayVisibility) {
		//record result to texture only when ray is visible
		//mainly to save memory bandwidth, but for stencil buffer we preserve the original stencil if stencil test fails
		return;
	}
	//store stencil result
	textureData.SSStencil[texIndex] = static_cast<unsigned char>(stencil_result | data.PrimitiveID);

	if (data.PrimitiveID == STPScreenSpaceRayIntersectionData::EnvironmentRayID) {
		//environment ray has no vertex data
		return;
	}
	const float3 pixel_position = ray_ori + ray_dir * data.RayTime;
	const uint2 pixel_uv = make_uint2(data.UV * cuda::std::numeric_limits<unsigned short>::max());
	surf2Dwrite(make_float4(pixel_position, 1.0f), textureData.GPosition, texCoord.x * sizeof(float4), texCoord.y);
	surf2Dwrite(make_ushort2(pixel_uv.x, pixel_uv.y), textureData.GTextureCoordinate, texCoord.x * sizeof(ushort2), texCoord.y);
}

__global__ void __closesthit__recordPrimitiveIntersection() {
	const auto& data = *reinterpret_cast<const STPScreenSpaceRayIntersectionData::STPPrimitiveHitData*>(optixGetSbtDataPointer()); 
	//read primitive vertex data
	const auto [objectID, instanceID] = STPInstanceIDCoder::decode(optixGetInstanceId());
	const STPGeometryAttributeFormat::STPIndexFormat& attributeIdx =
		data.PrimitiveIndex[objectID][instanceID][optixGetPrimitiveIndex()];
	//grab data of each vertex
	float2 uv[3];
	const auto* const baseVertex = data.PrimitiveVertex[objectID][instanceID];
	for (unsigned int i = 0u; i < 3u; i++) {
		const auto* const vertex = baseVertex + getByIndex(attributeIdx, i);
		
		uv[i] = vertex->UV;
	}

	//vertex interpolation
	STPSSRIPayload prd;
	const float2 bary = optixGetTriangleBarycentrics();
	prd.PrimitiveID = objectID;
	//we can recover vertex position from ray time
	prd.RayTime = optixGetRayTmax();
	prd.UV = STPFragmentUtility::barycentricInterpolation(bary, uv[0], uv[1], uv[2]);
	//store data to payload
	setPrimitiveData(prd);
}

__global__ void __miss__recordEnvironmentIntersection() {
	setEnvironmentData();
}