//THIS INLINE FILE IS MANAGED AUTOMATICALLY BY STPFreeSlipManager
#ifdef _STP_FREESLIP_MANAGER_CUH_

template<typename T>
__host__ SuperTerrainPlus::STPCompute::STPFreeSlipManager<T>::STPFreeSlipManager(T* texture, const STPFreeSlipData* data) :
	Data(data), Texture(texture) {

}

template<typename T>
__host__ SuperTerrainPlus::STPCompute::STPFreeSlipManager<T>::~STPFreeSlipManager() {

}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH T& SuperTerrainPlus::STPCompute::STPFreeSlipManager<T>::operator[](unsigned int global) {
	return const_cast<T&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH const T& SuperTerrainPlus::STPCompute::STPFreeSlipManager<T>::operator[](unsigned int global) const {
	return this->Texture[this->operator()(global)];
}

template<typename T>
MANAGER_HOST_DEVICE_SWITCH unsigned int SuperTerrainPlus::STPCompute::STPFreeSlipManager<T>::operator()(unsigned int global) const {
	return this->Data->GlobalLocalIndex[global];
}

#endif//_STP_FREESLIP_MANAGER_CUH_