//TEMPLATE DEFINITION FOR SMART DEVICE MEMORY
#ifdef _STP_SMART_DEVICE_MEMORY_H_

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

template<typename T>
inline void SuperTerrainPlus::STPSmartDeviceMemoryUtility::STPDeviceMemoryDeleter<T>::operator()(T* ptr) const {
	STPcudaCheckErr(cudaFree(ptr));
}

#endif//_STP_SMART_DEVICE_MEMORY_H_