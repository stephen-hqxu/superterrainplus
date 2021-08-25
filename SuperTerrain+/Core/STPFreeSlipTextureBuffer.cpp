#pragma once
#include <GPGPU/STPFreeSlipTextureBuffer.h>

//Error
#define STP_EXCEPTION_ON_ERROR
#include <SuperError+/STPDeviceErrorHandler.h>

#include <type_traits>
#include <stdexcept>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::rethrow_exception;
using std::current_exception;

static SuperTerrainPlus::STPRegularMemoryPool CallbackMemPool;

/**
 * @brief Essential data for stream-ordered host memory release
*/
struct STPHostReleaseData {
public:

	//Memory to be returned
	void* ReleasingHostMemory;
	//Memory pool where the memory will be returned to
	SuperTerrainPlus::STPPinnedMemoryPool* ReleasingHostMemPool;

};

//A CUDA stream callback function to release host memory
static void releaseHostMemory(void* release_data) {
	auto [memory, pool] = *reinterpret_cast<STPHostReleaseData*>(release_data);
	pool->release(memory);

	//free-up memory
	CallbackMemPool.release(release_data);
}

template<typename T>
STPFreeSlipTextureBuffer<T>::STPFreeSlipTextureBuffer(typename STPFreeSlipTexture& texture, const STPFreeSlipTextureData& data, const STPFreeSlipTextureAttribute& attr) : 
	Buffer(texture), Data(data), Attr(attr) {
	if (this->Buffer.empty()) {
		throw std::invalid_argument("provided free-slip texture is empty");
	}
}

template<typename T>
STPFreeSlipTextureBuffer<T>::~STPFreeSlipTextureBuffer() {
	if (!this->Integration) {
		//no texture has been integrated, nothing to do
		return;
	}
	//else we need to destroy allocation
	this->destroyAllocation();
}

template<typename T>
void STPFreeSlipTextureBuffer<T>::destroyAllocation() {
	//we can guarantee Integration is available
	auto [host, device, location] = this->Integration.value();
	const size_t pixel_per_chunk = this->Attr.TexturePixel * this->Data.Channel;
	const size_t freeslip_size = sizeof(T) * this->Buffer.size() * pixel_per_chunk;

	//we don't need to catch exception in destructor since it's dangerous to let the program keep running!!!
	//just let the program crash
	if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::ReadOnly) {
		//we need to copy the large buffer back to each chunk
		if (location == STPFreeSlipLocation::DeviceMemory) {
			//copy device memory to pinned memory we have allocated previously
			STPcudaCheckErr(cudaMemcpyAsync(host, device, freeslip_size, cudaMemcpyDeviceToHost, this->Data.Stream));
		}

		//disintegrate merged buffer
		{
			T* host_accumulator = host;
			for (T* map : this->Buffer) {
				STPcudaCheckErr(cudaMemcpyAsync(map, host_accumulator, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Data.Stream));
				host_accumulator += pixel_per_chunk;
			}
		}
	}
	//we don't need to copy the texture back to the original buffer if it's read only

	//deallocation
	//host memory will always be allocated
	//we need to add a callback because we don't want the host memory to be released right now as unfinished works may be still using it.
	STPHostReleaseData* release_data = static_cast<STPHostReleaseData*>(CallbackMemPool.request(sizeof(STPHostReleaseData)));
	release_data->ReleasingHostMemory = static_cast<void*>(host);
	release_data->ReleasingHostMemPool = &this->Attr.HostMemPool;
	STPcudaCheckErr(cudaLaunchHostFunc(this->Data.Stream, &releaseHostMemory, static_cast<void*>(release_data)));
	//if texture is in host the pointer is the same as PinnedMemoryBuffer
	if (location == STPFreeSlipLocation::DeviceMemory) {
		STPcudaCheckErr(cudaFreeAsync(device, this->Data.Stream));
	}
}

template<typename T>
T* STPFreeSlipTextureBuffer<T>::operator()(STPFreeSlipLocation location) {
	if (this->Integration) {
		//if we have already allocation with specified location, return directly
		auto [host, device, prev_location] = this->Integration.value();
		switch (prev_location) {
		case STPFreeSlipLocation::HostMemory:
			//only host memory is available
			return host;
			break;
		case STPFreeSlipLocation::DeviceMemory:
			//both host and device memory are available
			return (location == STPFreeSlipLocation::HostMemory) ? host : device;
			break;
		default:
			//impossible
			return nullptr;
			break;
		}
	}

	T* host, *device, *texture;
	const size_t pixel_per_chunk = this->Attr.TexturePixel * this->Data.Channel;
	const size_t freeslip_size = sizeof(T) * this->Buffer.size() * pixel_per_chunk;

	try {
		//we need host memory anyway
		//pinned memory will be needed anyway
		host = static_cast<T*>(this->Attr.HostMemPool.request(freeslip_size));
		if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::WriteOnly) {
			//make a initial copy from the original buffer if it's not write only
			//combine texture from each chunk to a large buffer
			T* host_accumulator = host;
			for (const T* map : this->Buffer) {
				STPcudaCheckErr(cudaMemcpyAsync(host_accumulator, map, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Data.Stream));
				host_accumulator += pixel_per_chunk;
			}
		}

		switch (location) {
		case STPFreeSlipLocation::HostMemory:
			//free-slip manager requires texture for host
			//we have got the host memory loaded
			texture = host;
			break;
		case STPFreeSlipLocation::DeviceMemory:
			//free-slip manager requires texture for device
			//device memory allocation
			STPcudaCheckErr(cudaMallocFromPoolAsync(&device, freeslip_size, this->Attr.DeviceMemPool, this->Data.Stream));
			//copy
			if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::WriteOnly) {
				STPcudaCheckErr(cudaMemcpyAsync(device, host, freeslip_size, cudaMemcpyHostToDevice, this->Data.Stream));
			}
			//no copy is needed if we only write to the buffer
			texture = device;
			break;
		default:
			texture = nullptr;
			break;
		}
	}
	catch (...) {
		rethrow_exception(current_exception());
	}

	//record the state
	this->Integration.emplace(host, device, location);
	//finally return merged buffer
	return texture;
}

template<typename T>
STPFreeSlipTextureBuffer<T>::operator STPFreeSlipLocation() const {
	if (!this->Integration) {
		throw std::logic_error("no memory location has been specified as no memory has been allocated");
	}
	return std::get<2>(this->Integration.value());
}

//Export Explicit Instantiation of Class
template class STP_API STPFreeSlipTextureBuffer<float>;
template class STP_API STPFreeSlipTextureBuffer<Sample>;
template class STP_API STPFreeSlipTextureBuffer<std::enable_if<!std::is_same<unsigned short, Sample>::value, unsigned short>>;