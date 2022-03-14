#include <SuperTerrain+/World/Chunk/FreeSlip/STPFreeSlipTextureBuffer.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
#include <SuperTerrain+/Exception/STPCUDAError.h>

//Template Def
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.tpp>

#include <type_traits>

#include <iostream>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::make_pair;
using std::unique_ptr;

template<typename T>
STPFreeSlipTextureBuffer<T>::STPHostCallbackDeleter::STPHostCallbackDeleter(cudaStream_t stream, STPPinnedMemoryPool* memPool) : Data(make_pair(stream, memPool)) {

}

template<typename T>
void STPFreeSlipTextureBuffer<T>::STPHostCallbackDeleter::operator()(T* ptr) const {
	if (!this->Data) {
		//nothing needs to be done if there's no data to free
		throw STPException::STPMemoryError("Memory free destination is not specified");
	}

	auto [stream, pool] = *this->Data;
	//deallocation
	//host memory will always be allocated
	//we need to wait for the stream because we don't want the host memory to be released right now as unfinished works may be still using it.
	//all pointers we provide are guaranteed to be valid until the stream has synced.
	STPcudaCheckErr(cudaStreamSynchronize(stream));
	pool->release(ptr);
	//if texture is in host the pointer is the same as PinnedMemoryBuffer
}

template<typename T>
STPFreeSlipTextureBuffer<T>::STPFreeSlipTextureBuffer(STPFreeSlipTexture& texture, STPFreeSlipTextureData data, const STPFreeSlipTextureAttribute& attr) : 
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
	const size_t pixel_per_chunk = this->Attr.TexturePixel * this->Data.Channel;
	const size_t freeslip_size = sizeof(T) * this->Buffer.size() * pixel_per_chunk;

	//we don't need to catch exception in destructor since it's dangerous to let the program keep running!!!
	//just let the program crash
	if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::ReadOnly) {
		//we need to copy the large buffer back to each chunk
		if (*this->Integration == STPFreeSlipLocation::DeviceMemory) {
			//copy device memory to pinned memory we have allocated previously
			STPcudaCheckErr(cudaMemcpyAsync(this->HostIntegration.get(), this->DeviceIntegration.get(), freeslip_size, cudaMemcpyDeviceToHost, this->Data.Stream));
		}

		//disintegrate merged buffer
		{
			T* host_accumulator = this->HostIntegration.get();
			for (T* map : this->Buffer) {
				STPcudaCheckErr(cudaMemcpyAsync(map, host_accumulator, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Data.Stream));
				host_accumulator += pixel_per_chunk;
			}
		}
	}
	//we don't need to copy the texture back to the original buffer if it's read only

	//deallocation
	//all memory will be freed by a streamed smart pointer
}

template<typename T>
T* STPFreeSlipTextureBuffer<T>::operator()(STPFreeSlipLocation location) {
	if (this->Integration) {
		//if we have already allocation with specified location, return directly
		switch (*this->Integration) {
		case STPFreeSlipLocation::HostMemory:
			//only host memory is available
			return this->HostIntegration.get();
			break;
		case STPFreeSlipLocation::DeviceMemory:
			//both host and device memory are available
			return (location == STPFreeSlipLocation::HostMemory) ? this->HostIntegration.get() : this->DeviceIntegration.get();
			break;
		default:
			//impossible
			throw STPException::STPInvalidArgument("The argument of memory location is not recognised");
			break;
		}
	}

	T* texture;
	const size_t pixel_per_chunk = this->Attr.TexturePixel * this->Data.Channel;
	const size_t freeslip_count = this->Buffer.size() * pixel_per_chunk,
		freeslip_size = sizeof(T) * freeslip_count;

	//we need host memory anyway
	//pinned memory will be needed anyway
	this->HostIntegration = unique_ptr<T[], STPHostCallbackDeleter>(
		static_cast<T*>(this->Attr.HostMemPool.request(freeslip_size)),
		STPHostCallbackDeleter(this->Data.Stream, &this->Attr.HostMemPool)
	);
	if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::WriteOnly) {
		//make a initial copy from the original buffer if it's not write only
		//combine texture from each chunk to a large buffer
		T* host_accumulator = this->HostIntegration.get();
		for (const T* map : this->Buffer) {
			STPcudaCheckErr(cudaMemcpyAsync(host_accumulator, map, pixel_per_chunk * sizeof(T), cudaMemcpyHostToHost, this->Data.Stream));
			host_accumulator += pixel_per_chunk;
		}
	}

	switch (location) {
	case STPFreeSlipLocation::HostMemory:
		//free-slip manager requires texture for host
		//we have got the host memory loaded
		texture = this->HostIntegration.get();
		break;
	case STPFreeSlipLocation::DeviceMemory:
		//free-slip manager requires texture for device
		//device memory allocation
		this->DeviceIntegration = STPSmartDeviceMemory::makeStreamedDevice<T[]>(this->Attr.DeviceMemPool, this->Data.Stream, freeslip_count);
		//copy
		if (this->Data.Mode != STPFreeSlipTextureData::STPMemoryMode::WriteOnly) {
			STPcudaCheckErr(cudaMemcpyAsync(this->DeviceIntegration.get(), this->HostIntegration.get(), freeslip_size, cudaMemcpyHostToDevice, this->Data.Stream));
		}
		//no copy is needed if we only write to the buffer
		texture = this->DeviceIntegration.get();
		break;
	default:
		throw STPException::STPInvalidArgument("The argument of memory location is not recognised");
		break;
	}

	//record the state
	this->Integration.emplace(location);
	//finally return merged buffer
	return texture;
}

template<typename T>
STPFreeSlipLocation STPFreeSlipTextureBuffer<T>::where() const {
	if (!this->Integration) {
		throw STPException::STPMemoryError("no memory location has been specified as no memory has been allocated");
	}
	return *this->Integration;
}

//Export Explicit Instantiation of Class
template class STP_API STPFreeSlipTextureBuffer<float>;
template class STP_API STPFreeSlipTextureBuffer<Sample>;
template class STP_API STPFreeSlipTextureBuffer<std::enable_if<!std::is_same<unsigned short, Sample>::value, unsigned short>>;