#pragma once
#include <SuperAlgorithm+/STPSimplexNoiseRTC.cuh>

#include <cuda_runtime.h>

using namespace SuperTerrainPlus::STPCompute;

__host__ __device__ STPSimplexNoiseRTC::STPSimplexNoiseRTC(const STPSimplexNoise* impl) : Impl(impl) {
	
}

__host__ __device__ STPSimplexNoiseRTC::~STPSimplexNoiseRTC() {

}

__device__ float STPSimplexNoiseRTC::simplex2D(float x, float y) const {
	return this->Impl->simplex2D(x, y);
}