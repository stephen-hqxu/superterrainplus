#include <SuperAlgorithm+/Device/STPKernelMath.cuh>

using namespace SuperTerrainPlus::STPCompute;

__device__ float STPKernelMath::lerp(float p1, float p2, float factor){
	factor = __saturatef(factor);
	return p1 * (1.0f - factor) + p2 * factor;
}

__device__ float STPKernelMath::Invlerp(float minVal, float maxVal, float value) {
	return __saturatef((value - minVal) / (maxVal - minVal));
}

__device__ float STPKernelMath::cosrp(float p1, float p2, float factor){
	factor = __saturatef(factor);
	const float cos_factor = (1.0f - cospif(factor)) / 2.0f;
	return STPKernelMath::lerp(p1, p2, cos_factor);
}

__device__ float STPKernelMath::clamp(float value, float min, float max){
	return fmaxf(min, fminf(value, max));
}