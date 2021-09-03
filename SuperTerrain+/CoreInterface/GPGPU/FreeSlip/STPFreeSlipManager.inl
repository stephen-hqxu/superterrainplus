#pragma once

//THIS INLINE FILE IS MANAGED AUTOMATICALLY BY STPFreeSlipManager
#ifdef _STP_FREESLIP_MANAGER_CUH_

#include "STPFreeSlipManager.cuh"

#include <type_traits>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

template<typename T>
__host__ STPFreeSlipManager<T>::STPFreeSlipManager(T* texture, const STPFreeSlipData* data) :
	Data(data), Texture(texture) {

}

template<typename T>
__host__ STPFreeSlipManager<T>::~STPFreeSlipManager() {

}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH T& STPFreeSlipManager<T>::operator[](unsigned int global) {
	return const_cast<T&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH const T& STPFreeSlipManager<T>::operator[](unsigned int global) const {
	return this->Texture[this->operator()(global)];
}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH unsigned int STPFreeSlipManager<T>::operator()(unsigned int global) const {
	return this->Data->GlobalLocalIndex == nullptr ? global : this->Data->GlobalLocalIndex[global];
}

#endif//_STP_FREESLIP_MANAGER_CUH_