#include "./Script/STPCommonGenerator.cuh"

//This is the dimension of map of one chunk
__constant__ uint2 Dimension[1];
//Dimension / 2
__constant__ float2 HalfDimension[1];
//This is the dimension of map in the entire rendered chunk
__constant__ uint2 RenderedDimension[1];

__device__ const uint2& STPCommonGenerator::mapDimension() {
	return *Dimension;
}

__device__ const float2& STPCommonGenerator::mapDimensionHalf() {
	return *HalfDimension;
}

__device__ const uint2& STPCommonGenerator::mapDimensionRendered() {
	return *RenderedDimension;
}