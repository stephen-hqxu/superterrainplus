#pragma once
#include <GPGPU/STPFreeSlipManager.cuh>

#include <type_traits>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

//free-slip data is copied
template<typename T>
__host__ STPFreeSlipManager<T>::STPFreeSlipManager(T* texture, const STPFreeSlipData* data) :
	Data(data), Texture(texture) {

}

template<typename T>
__host__ STPFreeSlipManager<T>::~STPFreeSlipManager() {

}

template<typename T>
__device__ __host__ T& STPFreeSlipManager<T>::operator[](unsigned int global) {
	return const_cast<T&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

template<typename T>
__device__ __host__ const T& STPFreeSlipManager<T>::operator[](unsigned int global) const {
	return this->Texture[this->operator()(global)];
}

template<typename T>
__device__ __host__ unsigned int STPFreeSlipManager<T>::operator()(unsigned int global) const {
	return this->Data->GlobalLocalIndex == nullptr ? global : this->Data->GlobalLocalIndex[global];
}

//Export explicit instantiations
template class STP_API STPFreeSlipManager<float>;
template class STP_API STPFreeSlipManager<Sample>;
template class STP_API STPFreeSlipManager<std::enable_if<!std::is_same<unsigned short, Sample>::value, unsigned short>>;