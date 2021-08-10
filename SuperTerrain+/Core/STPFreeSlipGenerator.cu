#pragma once
#include <GPGPU/STPFreeSlipGenerator.cuh>
#include <device_launch_parameters.h>

//Error
#define STP_EXCEPTION_ON_ERROR
#include <SuperError+/STPDeviceErrorHandler.h>

using namespace SuperTerrainPlus::STPCompute;

/**
 * @brief Generate a new global to local index table
 * @param output The generated table. Should be preallocated with size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y
 * @param rowCount The number of row in the global index table, which is equivalent to chunkRange.x * mapSize.x
 * @param chunkRange The number of chunk (or locals)
 * @param tableSize The x,y dimension of the table
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uint2, uint2, uint2);

STPFreeSlipGenerator::STPFreeSlipGenerator(uint2 range, uint2 mapSize) {
	this->Dimension = mapSize;
	this->FreeSlipChunk = range;
	this->FreeSlipRange = make_uint2(range.x * mapSize.x, range.y * mapSize.y);
	//set global local index
	this->initLocalGlobalIndexCUDA();
}

STPFreeSlipGenerator::~STPFreeSlipGenerator() {
	if (this->GlobalLocalIndex != nullptr) {
		STPcudaCheckErr(cudaFree(this->GlobalLocalIndex));
	}
}

__host__ void STPFreeSlipGenerator::initLocalGlobalIndexCUDA() {
	const uint2& global_dimension = this->FreeSlipRange;
	//launch parameters
	int Mingridsize, blocksize;
	dim3 Dimgridsize, Dimblocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &initGlobalLocalIndexKERNEL));
	Dimblocksize = dim3(32, blocksize / 32);
	Dimgridsize = dim3((global_dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (global_dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

	//Don't generate the table when FreeSlipChunk.xy are both 1, and in STPRainDrop don't use the table
	if (this->FreeSlipChunk.x == 1u && this->FreeSlipChunk.y == 1u) {
		this->GlobalLocalIndex = nullptr;
		return;
	}

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//allocation
	STPcudaCheckErr(cudaMalloc(&this->GlobalLocalIndex, sizeof(unsigned int) * global_dimension.x * global_dimension.y));
	//compute
	initGlobalLocalIndexKERNEL << <Dimgridsize, Dimblocksize >> > (this->GlobalLocalIndex, global_dimension.x, this->FreeSlipChunk, global_dimension, this->Dimension);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());
}

__host__ const uint2& STPFreeSlipGenerator::getDimension() const {
	return this->Dimension;
}

__host__ const uint2& STPFreeSlipGenerator::getFreeSlipChunk() const {
	return this->FreeSlipChunk;
}

__host__ const uint2& STPFreeSlipGenerator::getFreeSlipRange() const {
	return this->FreeSlipRange;
}

__host__ STPFreeSlipManager STPFreeSlipGenerator::getManager(float* texture) const {
	return STPFreeSlipManager(texture, dynamic_cast<const STPFreeSlipData*>(this));
}

__global__ void initGlobalLocalIndexKERNEL(unsigned int* output, unsigned int rowCount, uint2 chunkRange, uint2 tableSize, uint2 mapSize) {
	//current pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		globalidx = x + y * rowCount;
	if (x >= tableSize.x || y >= tableSize.y) {
		return;
	}

	//simple maths
	const uint2 globalPos = make_uint2(globalidx - floorf(globalidx / rowCount) * rowCount, floorf(globalidx / rowCount));
	const uint2 chunkPos = make_uint2(floorf(globalPos.x / mapSize.x), floorf(globalPos.y / mapSize.y));
	const uint2 localPos = make_uint2(globalPos.x - chunkPos.x * mapSize.x, globalPos.y - chunkPos.y * mapSize.y);

	output[globalidx] = (chunkPos.x + chunkRange.x * chunkPos.y) * mapSize.x * mapSize.y + (localPos.x + mapSize.x * localPos.y);
}