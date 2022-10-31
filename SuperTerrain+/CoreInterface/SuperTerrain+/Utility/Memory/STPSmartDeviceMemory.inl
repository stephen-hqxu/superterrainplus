//TEMPLATE DEFINITION FOR SMART DEVICE MEMORY
#ifdef _STP_SMART_DEVICE_MEMORY_H_

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

template<typename T>
inline void SuperTerrainPlus::STPSmartDeviceMemory::STPSmartDeviceMemoryImpl::STPPinnedMemoryDeleter<T>::operator()(T* ptr) const {
	STP_CHECK_CUDA(cudaFreeHost(ptr));
}

template<typename T>
inline void SuperTerrainPlus::STPSmartDeviceMemory::STPSmartDeviceMemoryImpl::STPDeviceMemoryDeleter<T>::operator()(T* ptr) const {
	STP_CHECK_CUDA(cudaFree(ptr));
}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPSmartDeviceMemoryImpl::STPStreamedDeviceMemoryDeleter<T>::STPStreamedDeviceMemoryDeleter
	(cudaStream_t stream) : Stream(stream) {
	
}

template<typename T>
inline void SuperTerrainPlus::STPSmartDeviceMemory::STPSmartDeviceMemoryImpl::STPStreamedDeviceMemoryDeleter<T>::operator()
	(T* ptr) const {
	STP_CHECK_CUDA(cudaFreeAsync(ptr, *this->Stream));
}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPPitchedDeviceMemory<T>::STPPitchedDeviceMemory() : Pitch(0u) {

}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPPitchedDeviceMemory<T>::STPPitchedDeviceMemory
	(STPSmartDeviceMemoryImpl::NoArray<T>* ptr, size_t pitch) :
	STPDeviceMemory<T>(ptr), Pitch(pitch) {
	
}

#define TYPE_SANITISE using U = STPSmartDeviceMemoryImpl::NoArray<T>; \
U* cache

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPPinnedMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makePinned(size_t size) {
	TYPE_SANITISE;

	STP_CHECK_CUDA(cudaMallocHost(&cache, sizeof(U) * size));
	return STPSmartDeviceMemory::STPPinnedMemory<T>(cache);
}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPDeviceMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makeDevice(size_t size) {
	TYPE_SANITISE;

	//remember size denotes the number of element
	STP_CHECK_CUDA(cudaMalloc(&cache, sizeof(U) * size));
	//if any exception is thrown during malloc, it will not proceed
	//exception thrown at malloc will prevent any memory to be allocated, so we don't need to free it.
	return STPDeviceMemory<T>(cache);
}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPStreamedDeviceMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makeStreamedDevice
	(cudaMemPool_t memPool, cudaStream_t stream, size_t size) {
	TYPE_SANITISE;

	//allocate using the pool
	STP_CHECK_CUDA(cudaMallocFromPoolAsync(&cache, sizeof(U) * size, memPool, stream));
	//init the streamed deleter
	return STPStreamedDeviceMemory<T>(cache, STPSmartDeviceMemoryImpl::STPStreamedDeviceMemoryDeleter<U>(stream));
}

template<typename T>
inline SuperTerrainPlus::STPSmartDeviceMemory::STPPitchedDeviceMemory<T> SuperTerrainPlus::STPSmartDeviceMemory::makePitchedDevice(
	size_t width, size_t height) {
	TYPE_SANITISE;

	size_t pitch;
	STP_CHECK_CUDA(cudaMallocPitch(&cache, &pitch, sizeof(U) * width, height));
	return STPPitchedDeviceMemory<T>(cache, pitch);
}

#undef TYPE_SANITISE

#endif//_STP_SMART_DEVICE_MEMORY_H_