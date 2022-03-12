//TEMPLATE DEFINITION FOR SMART DEVICE MEMORY
#ifdef _STP_SMART_DEVICE_MEMORY_H_

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

template<typename T>
void SuperTerrainPlus::STPSmartDeviceMemory::STPDeviceMemoryDeleter<T>::operator()(T* ptr) const {
	STPcudaCheckErr(cudaFree(ptr));
}

template<typename T>
SuperTerrainPlus::STPSmartDeviceMemory::STPStreamedDeviceMemoryDeleter<T>::STPStreamedDeviceMemoryDeleter(cudaStream_t stream) : Stream(stream) {
	
}

template<typename T>
void SuperTerrainPlus::STPSmartDeviceMemory::STPStreamedDeviceMemoryDeleter<T>::operator()(T* ptr) const {
	if (!this->Stream) {
		//check if user has assigned a stream
		throw SuperTerrainPlus::STPException::STPMemoryError("Invoking a deleter with no CUDA stream assigned");
	}
	STPcudaCheckErr(cudaFreeAsync(ptr, *this->Stream));
}

template<typename T>
SuperTerrainPlus::STPSmartDeviceMemory::STPDeviceMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makeDevice(size_t size) {
	//type sanitise
	using U = typename NoArray<T>;
	
	U* cache;
	//remember size denotes the number of element
	STPcudaCheckErr(cudaMalloc(&cache, sizeof(U) * size));
	//if any exception is thrown during malloc, it will not proceed
	//exception thrown at malloc will prevent any memory to be allocated, so we don't need to free it.
	return STPDeviceMemory<T>(cache);
}

template<typename T>
SuperTerrainPlus::STPSmartDeviceMemory::STPStreamedDeviceMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makeStreamedDevice(cudaMemPool_t memPool, cudaStream_t stream, size_t size) {
	//type sanitise
	using U = typename NoArray<T>;
	
	U* cache;
	//allocate using the pool
	STPcudaCheckErr(cudaMallocFromPoolAsync(&cache, sizeof(U) * size, memPool, stream));
	//init the streamed deleter
	return STPStreamedDeviceMemory<T>(cache, STPStreamedDeviceMemoryDeleter<U>(stream));
}

#endif//_STP_SMART_DEVICE_MEMORY_H_