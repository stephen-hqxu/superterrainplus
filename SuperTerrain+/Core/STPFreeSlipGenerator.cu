#pragma once
#include <GPGPU/STPFreeSlipGenerator.cuh>
#include <device_launch_parameters.h>

#include <type_traits>
#include <stdexcept>

//Error
#define STP_EXCEPTION_ON_ERROR
#include <Utility/STPDeviceErrorHandler.h>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::make_unique;
using std::exception;
using std::rethrow_exception;
using std::current_exception;

#define SampleIsUint16 std::enable_if<!std::is_same<unsigned short, Sample>::value, unsigned short>

/**
 * @brief Generate a new global to local index table
 * @param output The generated table. Should be preallocated with size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y
 * @param rowCount The number of row in the global index table, which is equivalent to chunkRange.x * mapSize.x
 * @param chunkRange The number of chunk (or locals)
 * @param tableSize The x,y dimension of the table
 * @param mapSize The dimension of the map
*/
__global__ void initGlobalLocalIndexKERNEL(unsigned int*, unsigned int, uint2, uint2, uint2);

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
		data = this->Generator.Data_Device;
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
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<SampleIsUint16>;

__host__ STPFreeSlipGenerator::STPFreeSlipGenerator(uint2 range, uint2 mapSize) : 
	STPFreeSlipData{ nullptr, mapSize, range, make_uint2(range.x * mapSize.x, range.y * mapSize.y) } {
	try {
		//set global local index
		this->initLocalGlobalIndexCUDA();
	}
	catch (...) {
		rethrow_exception(current_exception());
	}
}

__host__ STPFreeSlipGenerator::~STPFreeSlipGenerator() {
	this->clearDeviceIndex();
}

__host__ void STPFreeSlipGenerator::initLocalGlobalIndexCUDA() {
	const uint2& global_dimension = this->FreeSlipRange;
	const size_t index_count = global_dimension.x * global_dimension.y;
	try {
		//launch parameters
		int Mingridsize, blocksize;
		dim3 Dimgridsize, Dimblocksize;
		STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &initGlobalLocalIndexKERNEL));
		Dimblocksize = dim3(32, blocksize / 32);
		Dimgridsize = dim3((global_dimension.x + Dimblocksize.x - 1) / Dimblocksize.x, (global_dimension.y + Dimblocksize.y - 1) / Dimblocksize.y);

		//Don't generate the table when FreeSlipChunk.xy are both 1, and in STPRainDrop don't use the table
		if (this->FreeSlipChunk.x == 1u && this->FreeSlipChunk.y == 1u) {
			this->Index_Device = nullptr;
			return;
		}

		//make sure all previous takes are finished
		STPcudaCheckErr(cudaDeviceSynchronize());
		//allocation
		STPcudaCheckErr(cudaMalloc(&this->Index_Device, sizeof(unsigned int) * index_count));
		//compute
		initGlobalLocalIndexKERNEL << <Dimgridsize, Dimblocksize >> > (this->Index_Device, global_dimension.x, this->FreeSlipChunk, global_dimension, this->Dimension);
		STPcudaCheckErr(cudaGetLastError());
		STPcudaCheckErr(cudaDeviceSynchronize());

		//make a copy of index table on host
		//the copy that the generator inherited is a host copy, the host pointer is managed by unique_ptr
		this->Index_Host = make_unique<unsigned int[]>(index_count);
		STPcudaCheckErr(cudaMemcpy(this->Index_Host.get(), this->Index_Device, sizeof(unsigned int) * index_count, cudaMemcpyDeviceToHost));
		this->GlobalLocalIndex = this->Index_Host.get();
		//get a device version of free-slip data
		STPFreeSlipData device_buffer(dynamic_cast<const STPFreeSlipData&>(*this));
		device_buffer.GlobalLocalIndex = this->Index_Device;
		STPcudaCheckErr(cudaMalloc(&this->Data_Device, sizeof(STPFreeSlipData)));
		STPcudaCheckErr(cudaMemcpy(this->Data_Device, &device_buffer, sizeof(STPFreeSlipData), cudaMemcpyHostToDevice));
	}
	catch (const exception& e) {
		//clear device memory (if any) to avoid memory leaks
		this->clearDeviceIndex();
		throw e;
	}
}

__host__ void STPFreeSlipGenerator::clearDeviceIndex() noexcept {
	if (this->Data_Device != nullptr) {
		STPcudaCheckErr(cudaFree(this->Data_Device));
	}
	if (this->Index_Device != nullptr) {
		STPcudaCheckErr(cudaFree(this->Index_Device));
	}
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

#define getManagerAdapter STPFreeSlipManagerAdaptor(buffer, *this)

template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<float> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<float>& buffer) const {
	return getManagerAdapter;
}

template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<Sample> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<Sample>& buffer) const {
	return getManagerAdapter;
}

template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<SampleIsUint16> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<SampleIsUint16>& buffer) const {
	return getManagerAdapter;
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