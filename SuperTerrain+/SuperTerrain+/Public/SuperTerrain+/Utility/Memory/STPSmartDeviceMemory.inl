//TEMPLATE DEFINITION FOR SMART DEVICE MEMORY
#ifdef _STP_SMART_DEVICE_MEMORY_H_

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

#define DEV_MEM_NAME SuperTerrainPlus::STPSmartDeviceMemory

template<typename T>
inline void DEV_MEM_NAME::STPImplementation::STPHostDeleter<T>::operator()(T* const ptr) const {
	STP_CHECK_CUDA(cudaFreeHost(ptr));
}

template<typename T>
inline void DEV_MEM_NAME::STPImplementation::STPDeviceDeleter<T>::operator()(T* const ptr) const {
	STP_CHECK_CUDA(cudaFree(ptr));
}

template<typename T>
inline DEV_MEM_NAME::STPImplementation::STPStreamedDeviceDeleter<T>::STPStreamedDeviceDeleter
	(const cudaStream_t stream) noexcept : Stream(stream) {
	
}

template<typename T>
inline void DEV_MEM_NAME::STPImplementation::STPStreamedDeviceDeleter<T>::operator()(T* const ptr) const {
	STP_CHECK_CUDA(cudaFreeAsync(ptr, this->Stream));
}

template<typename T>
inline DEV_MEM_NAME::STPPitchedDevice<T>::STPPitchedDevice() noexcept : Pitch(0u) {

}

template<typename T>
inline DEV_MEM_NAME::STPPitchedDevice<T>::STPPitchedDevice(const typename STPDevice<T>::pointer ptr, const size_t pitch) noexcept :
	STPDevice<T>(ptr), Pitch(pitch) {
	
}

#define TYPE_SANITISE(MEM_TYPE, VAR_NAME) using U = typename MEM_TYPE::element_type; \
typename MEM_TYPE::pointer VAR_NAME

template<typename T>
inline DEV_MEM_NAME::STPHost<T> DEV_MEM_NAME::makeHost(const size_t size) {
	TYPE_SANITISE(STPHost<T>, pinned_mem);

	STP_CHECK_CUDA(cudaMallocHost(&pinned_mem, sizeof(U) * size));
	return STPHost<T>(pinned_mem);
}

template<typename T>
inline DEV_MEM_NAME::STPDevice<T> DEV_MEM_NAME::makeDevice(const size_t size) {
	TYPE_SANITISE(STPDevice<T>, device_mem);

	//remember size denotes the number of element
	STP_CHECK_CUDA(cudaMalloc(&device_mem, sizeof(U) * size));
	//if any exception is thrown during malloc, it will not proceed
	//exception thrown at malloc will prevent any memory to be allocated, so we don't need to free it.
	return STPDevice<T>(device_mem);
}

template<typename T>
inline DEV_MEM_NAME::STPStreamedDevice<T> DEV_MEM_NAME::makeStreamedDevice(
	const cudaMemPool_t memPool, const cudaStream_t stream, const size_t size) {
	TYPE_SANITISE(STPStreamedDevice<T>, streamed_mem);

	//allocate using the pool
	STP_CHECK_CUDA(cudaMallocFromPoolAsync(&streamed_mem, sizeof(U) * size, memPool, stream));
	//init the streamed deleter
	return STPStreamedDevice<T>(streamed_mem, { stream });
}

template<typename T>
inline DEV_MEM_NAME::STPPitchedDevice<T> DEV_MEM_NAME::makePitchedDevice(
	const size_t width, const size_t height) {
	TYPE_SANITISE(STPPitchedDevice<T>, pitched_mem);

	size_t pitch;
	STP_CHECK_CUDA(cudaMallocPitch(&pitched_mem, &pitch, sizeof(U) * width, height));
	return STPPitchedDevice<T>(pitched_mem, pitch);
}

#undef TYPE_SANITISE
#undef DEV_MEM_NAME

#endif//_STP_SMART_DEVICE_MEMORY_H_