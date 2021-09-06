#pragma once
#include <SuperTerrain+/GPGPU/FreeSlip/STPFreeSlipTextureBuffer.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>

#include <type_traits>

#include <iostream>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

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
	STPHostReleaseData* data = reinterpret_cast<STPHostReleaseData*>(release_data);
	auto [memory, pool] = *data;
	pool->release(memory);

	//free-up memory
	if constexpr (!std::is_trivially_destructible_v<STPHostReleaseData>) {
		//we need to call the destructor if it has user-defined destructor before returning to the thread pool.
		data->~STPHostReleaseData();
	}
	CallbackMemPool.release(release_data);
}

template<typename T>
STPFreeSlipTextureBuffer<T>::STPFreeSlipTextureBuffer(typename STPFreeSlipTexture& texture, const STPFreeSlipTextureData& data, const STPFreeSlipTextureAttribute& attr) : 
	Buffer(texture), Data(data), Attr(attr) {
	if (this->Buffer.empty()) {
		throw STPException::STPInvalidArgument("Provided free-slip texture is empty");
	}
	if (this->Attr.TexturePixel == 0ull) {
		throw STPException::STPBadNumericRange("Number of pixel should be a positive integer");
	}
	if (this->Data.Channel == 0u) {
		throw STPException::STPBadNumericRange("Number of texture channel should be a positive integer");
	}
}

template<typename T>
STPFreeSlipTextureBuffer<T>::~STPFreeSlipTextureBuffer() noexcept(false) try {
	if (!this->Integration) {
		//no texture has been integrated, nothing to do
		return;
	}
	//else we need to destroy allocation
	this->destroyAllocation();
	
}
catch (const STPException::STPCUDAError& cuda_err) {
	using std::cerr;
	using std::endl;
	//if exception is caught during destroying memory, it's mostly caused by illegal memory access and is very dangerous if let go
	//simply terminate the program
	cerr << cuda_err.what() << endl;
	std::terminate();
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
	//all pointers we provide are guaranteed to be valid until the stream has synced.
	//use placement new to init STPHostReleaseData after allocating raw memory
	STPHostReleaseData* release_data = new(CallbackMemPool.request(sizeof(STPHostReleaseData))) STPHostReleaseData();
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
			throw STPException::STPInvalidArgument("The argument of memory location is not recognised");
			break;
		}
	}

	T* host = nullptr, *device = nullptr, *texture;
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
			throw STPException::STPInvalidArgument("The argument of memory location is not recognised");
			break;
		}
	}
	catch (...) {
		//clear up memory
		if (host != nullptr) {
			this->Attr.HostMemPool.release(host);
		}
		if (device != nullptr) {
			STPcudaCheckErr(cudaFreeAsync(device, this->Data.Stream));
		}

		throw;
	}

	//record the state
	this->Integration.emplace(host, device, location);
	//finally return merged buffer
	return texture;
}

template<typename T>
STPFreeSlipTextureBuffer<T>::operator STPFreeSlipLocation() const {
	if (!this->Integration) {
		throw STPException::STPMemoryError("no memory location has been specified as no memory has been allocated");
	}
	return std::get<2>(this->Integration.value());
}

//Export Explicit Instantiation of Class
template class STP_API STPFreeSlipTextureBuffer<float>;
template class STP_API STPFreeSlipTextureBuffer<Sample>;
template class STP_API STPFreeSlipTextureBuffer<std::enable_if<!std::is_same<unsigned short, Sample>::value, unsigned short>>;