#pragma once
#include <SuperTerrain+/GPGPU/FreeSlip/STPFreeSlipGenerator.cuh>
#include <device_launch_parameters.h>

#include <type_traits>
#include <stdexcept>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>

#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::make_unique;
using std::exception;

//GLM
#include <glm/geometric.hpp>

using glm::uvec2;

static_assert(std::is_same_v<unsigned short, Sample>, "Rendering buffer format is not exported");

/**
 * @brief Generate a new global to local index table
 * @param output The generated table. Should be preallocated with size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y
 * @param rowCount The number of row in the global index table, which is equivalent to chunkRange.x * mapSize.x
 * @param chunkRange The number of chunk (or locals)
 * @param tableSize The x,y dimension of the table
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uvec2, uvec2, uvec2);

template<typename T>
__host__ STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::STPFreeSlipManagerAdaptor(STPFreeSlipTextureBuffer<T>& buffer, const STPFreeSlipGenerator& generator) :
	Generator(generator), Texture(buffer) {
}

template<typename T>
__host__ STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::~STPFreeSlipManagerAdaptor() {
	
}

template<typename T>
__host__ STPFreeSlipManager<T> STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::operator()(STPFreeSlipLocation location) const {
	const STPFreeSlipData* data;
	T* texture = this->Texture(location);

	switch (location) {
	case STPFreeSlipLocation::HostMemory:
		//free-slip manager requires a host index table
		data = dynamic_cast<const STPFreeSlipData*>(&this->Generator);
		break;
	case STPFreeSlipLocation::DeviceMemory:
		//free-slip manager requires a device index table
		data = this->Generator.Data_Device.get();
		break;
	default:
		data = nullptr;
		break;
	}

	return STPFreeSlipManager<T>(texture, data);
}

//Export Instantiations of STPFreeSlipManagerAdaptor
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<float>;
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<Sample>;

__host__ STPFreeSlipGenerator::STPFreeSlipGenerator(uvec2 range, uvec2 mapSize) : 
	STPFreeSlipData{ nullptr, mapSize, range, range * mapSize } {
	if (this->FreeSlipRange.x == 0u || this->FreeSlipRange.y == 0u) {
		throw STPException::STPBadNumericRange("Dimension or/and free-slip chunk size should not be zero");
	}

	//set global local index
	this->initLocalGlobalIndexCUDA();
}

__host__ STPFreeSlipGenerator::~STPFreeSlipGenerator() {

}

__host__ void STPFreeSlipGenerator::initLocalGlobalIndexCUDA() {
	const uvec2& global_dimension = this->FreeSlipRange;
	const size_t index_count = global_dimension.x * global_dimension.y;
	
	//launch parameters
	int Mingridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &initGlobalLocalIndexKERNEL));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		Dimgridsize = (global_dimension + Dimblocksize - 1u) / Dimblocksize;

	//make sure all previous takes are finished
	STPcudaCheckErr(cudaDeviceSynchronize());
	//allocation
	this->Index_Device = STPSmartDeviceMemory::makeDevice<unsigned int[]>(index_count);
	//compute
	initGlobalLocalIndexKERNEL << <dim3(Dimgridsize.x, Dimgridsize.y), dim3(Dimblocksize.x, Dimblocksize.y) >> > (this->Index_Device.get(), global_dimension.x, this->FreeSlipChunk, global_dimension, this->Dimension);
	STPcudaCheckErr(cudaGetLastError());
	STPcudaCheckErr(cudaDeviceSynchronize());

	//make a copy of index table on host
	//the copy that the generator inherited is a host copy, the host pointer is managed by unique_ptr
	this->Index_Host = make_unique<unsigned int[]>(index_count);
	STPcudaCheckErr(cudaMemcpy(this->Index_Host.get(), this->Index_Device.get(), sizeof(unsigned int) * index_count, cudaMemcpyDeviceToHost));
	this->GlobalLocalIndex = this->Index_Host.get();
	//get a device version of free-slip data
	STPFreeSlipData device_buffer(dynamic_cast<const STPFreeSlipData&>(*this));
	device_buffer.GlobalLocalIndex = this->Index_Device.get();
	this->Data_Device = STPSmartDeviceMemory::makeDevice<STPFreeSlipData>();
	STPcudaCheckErr(cudaMemcpy(this->Data_Device.get(), &device_buffer, sizeof(STPFreeSlipData), cudaMemcpyHostToDevice));
}

__host__ const uvec2& STPFreeSlipGenerator::getDimension() const {
	return this->Dimension;
}

__host__ const uvec2& STPFreeSlipGenerator::getFreeSlipChunk() const {
	return this->FreeSlipChunk;
}

__host__ const uvec2& STPFreeSlipGenerator::getFreeSlipRange() const {
	return this->FreeSlipRange;
}

#define GET_MANAGER(TYPE) \
template<> \
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<TYPE> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<TYPE>& buffer) const { \
	return STPFreeSlipManagerAdaptor(buffer, *this); \
}

GET_MANAGER(float)

GET_MANAGER(Sample)

__global__ void initGlobalLocalIndexKERNEL(unsigned int* output, unsigned int rowCount, uvec2 chunkRange, uvec2 tableSize, uvec2 mapSize) {
	using glm::vec2;
	//current pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		globalidx = x + y * rowCount;
	if (x >= tableSize.x || y >= tableSize.y) {
		return;
	}

	//simple maths
	const uvec2 globalPos = uvec2(globalidx - y * rowCount, y);
	const uvec2 chunkPos = globalPos / mapSize;//non-negative integer division is a floor
	const uvec2 localPos = globalPos - chunkPos * mapSize;

	output[globalidx] = (chunkPos.x + chunkRange.x * chunkPos.y) * mapSize.x * mapSize.y + (localPos.x + mapSize.x * localPos.y);
}