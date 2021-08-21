#pragma once
#include <GPGPU/STPFreeSlipGenerator.cuh>
#include <device_launch_parameters.h>

#include <type_traits>
#include <stdexcept>

//Error
#define STP_EXCEPTION_ON_ERROR
#include <SuperError+/STPDeviceErrorHandler.h>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::make_unique;
using std::make_optional;
using std::tie;
using std::exception;
using std::rethrow_exception;
using std::current_exception;

constexpr static bool SampleIsUint16 = std::is_same<Sample, unsigned short>::value;

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
__host__ STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::STPFreeSlipManagerAdaptor(STPFreeSlipTexture& texture, const STPFreeSlipGenerator& generator, cudaStream_t stream) :
	Generator(generator), Buffer(texture), Stream(stream) {
	if (this->Buffer.empty()) {
		throw std::invalid_argument("provided free-slip texture is empty");
	}
}

template<typename T>
__host__ STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::~STPFreeSlipManagerAdaptor() {
	if (!this->Integration) {
		//no texture has been integrated, nothing to do
		return;
	}

	//disintegration buffer
	//we can guarantee Integration is available
	auto [texture, channel, location, read_only] = this->Integration.value();
	const size_t pixel_per_chunk = this->Generator.Dimension.x * this->Generator.Dimension.y * channel;
	const size_t freeslip_size = sizeof(T) * this->Buffer.size() * pixel_per_chunk;

	if (!read_only) {
		//we need to copy the large buffer back to each chunk
		if (location == STPFreeSlipLocation::DeviceMemory) {
			//copy device memory to pinned memory we have allocated previously
			STPcudaCheckErr(cudaMemcpyAsync(this->PinnedMemoryBuffer, texture, freeslip_size, cudaMemcpyDeviceToHost, this->Stream));
		}

		//disintegrate merged buffer
		{
			T* host_mem = this->PinnedMemoryBuffer;
			for (T* map : this->Buffer) {
				STPcudaCheckErr(cudaMemcpyAsync(map, host_mem, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Stream));
				host_mem += pixel_per_chunk;
			}
		}
	}

	//deallocation
	//host memory will always be allocated
	STPcudaCheckErr(cudaFreeHost(this->PinnedMemoryBuffer));
	//if texture is in host the pointer is the same as PinnedMemoryBuffer
	if (location == STPFreeSlipLocation::DeviceMemory) {
		STPcudaCheckErr(cudaFree(texture));
	}
}

template<typename T>
__host__ STPFreeSlipManager<T> STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::operator()(STPFreeSlipLocation location, bool read_only, unsigned char channel) const {
	T* texture;
	const STPFreeSlipData* data;
	const size_t pixel_per_chunk = this->Generator.Dimension.x * this->Generator.Dimension.y * channel;
	const size_t freeslip_size = sizeof(T) * this->Buffer.size() * pixel_per_chunk;

	//we need host memory anyway
	//pinned memory will be neede anyway
	STPcudaCheckErr(cudaMallocHost(&this->PinnedMemoryBuffer, freeslip_size));
	//combine texture from each chunk to a large buffer
	{
		T* host_mem = this->PinnedMemoryBuffer;
		for (const T* map : this->Buffer) {
			STPcudaCheckErr(cudaMemcpyAsync(host_mem, map, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Stream));
			host_mem += pixel_per_chunk;
		}
	}

	switch (location) {
	case STPFreeSlipLocation::HostMemory:
		//free-slip manager requires texture for host
		//we have got the host memory loaded
		texture = this->PinnedMemoryBuffer;

		//free-slip manager requires a host index table
		data = dynamic_cast<const STPFreeSlipData*>(&this->Generator);
		break;
	case STPFreeSlipLocation::DeviceMemory:
		//free-slip manager requires texture for device
		//device memory allocation
		STPcudaCheckErr(cudaMallocFromPoolAsync(&texture, freeslip_size, this->Generator.DevicePool, this->Stream));
		//copy
		STPcudaCheckErr(cudaMemcpyAsync(texture, this->PinnedMemoryBuffer, freeslip_size, cudaMemcpyHostToDevice, this->Stream));

		//free-slip manager requires a device index table
		data = this->Generator.Data_Device;
		break;
	default:
		texture = nullptr;
		data = nullptr;
		break;
	}

	//record the state
	this->Integration = make_optional(tie(texture, channel, location, read_only));
	//finally return free-slip manager
	return STPFreeSlipManager(texture, data);
}

//Export Instantiations of STPFreeSlipManagerAdaptor
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<float>;
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<Sample>;
#if not SampleIsUint16
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<unsigned short>;
#endif

__host__ STPFreeSlipGenerator::STPFreeSlipGenerator(uint2 range, uint2 mapSize) : DevicePool(0) {
	this->Dimension = mapSize;
	this->FreeSlipChunk = range;
	this->FreeSlipRange = make_uint2(range.x * mapSize.x, range.y * mapSize.y);
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

__host__ void STPFreeSlipGenerator::setDeviceMemPool(cudaMemPool_t device_mempool) {
	this->DevicePool = device_mempool;
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

#define getManagerAdapter STPFreeSlipManagerAdaptor(texture, *this, stream)

template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<float> STPFreeSlipGenerator::getAdaptor(STPFreeSlipManagerAdaptor<float>::STPFreeSlipTexture& texture, cudaStream_t stream) const {
	return getManagerAdapter;
}

template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<Sample> STPFreeSlipGenerator::getAdaptor(STPFreeSlipManagerAdaptor<Sample>::STPFreeSlipTexture& texture, cudaStream_t stream) const {
	return getManagerAdapter;
}

#if not SampleIsUint16
template<>
__host__ STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<unsigned short> STPFreeSlipGenerator::getAdaptor(STPFreeSlipManagerAdaptor<unsigned short>::STPFreeSlipTexture& texture, cudaStream_t stream) const {
	return getManagerAdapter;
}
#endif

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