#include <SuperTerrain+/World/Chunk/FreeSlip/STPFreeSlipGenerator.h>
#include <device_launch_parameters.h>

#include <type_traits>
#include <stdexcept>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.tpp>

//Kernel Implementation
#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>
//CUDA
#include <cuda_runtime.h>

using namespace SuperTerrainPlus::STPCompute;
using SuperTerrainPlus::STPDiversity::Sample;

using std::make_unique;
using std::exception;

//GLM
#include <glm/geometric.hpp>

using glm::uvec2;

static_assert(std::is_same_v<unsigned short, Sample>, "Rendering buffer format is not exported");

template<typename T>
STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::STPFreeSlipManagerAdaptor(STPFreeSlipTextureBuffer<T>& buffer, const STPFreeSlipGenerator& generator) :
	Generator(generator), Texture(buffer) {
}

template<typename T>
STPFreeSlipManager<T> STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T>::operator()(STPFreeSlipLocation location) const {
	const STPFreeSlipData* data;
	T* texture = this->Texture(location);

	switch (location) {
	case STPFreeSlipLocation::HostMemory:
		//free-slip manager requires a host index table
		data = &this->Generator.Data;
		break;
	case STPFreeSlipLocation::DeviceMemory:
		//free-slip manager requires a device index table
		data = this->Generator.Data_Device.get();
		break;
	default:
		data = nullptr;
		break;
	}

	return STPFreeSlipManager<T>(texture, data);
}

//Export Instantiations of STPFreeSlipManagerAdaptor
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<float>;
template class STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<Sample>;

STPFreeSlipGenerator::STPFreeSlipGenerator(uvec2 range, uvec2 mapSize) : 
	Data{ nullptr, mapSize, range, range * mapSize } {
	if (this->Data.FreeSlipRange.x == 0u || this->Data.FreeSlipRange.y == 0u) {
		throw STPException::STPBadNumericRange("Dimension or/and free-slip chunk size should not be zero");
	}

	//set global local index
	const uvec2& slipRange = this->Data.FreeSlipRange;
	this->Index_Device = STPHeightfieldKernel::initGlobalLocalIndex(
		range, slipRange, mapSize
	);

	const unsigned int index_count = slipRange.x * slipRange.y;
	//make a copy of index table on host
	//the copy that the generator inherited is a host copy, the host pointer is managed by unique_ptr
	this->Index_Host = make_unique<unsigned int[]>(index_count);
	STPcudaCheckErr(cudaMemcpy(this->Index_Host.get(), this->Index_Device.get(), sizeof(unsigned int) * index_count, cudaMemcpyDeviceToHost));
	this->Data.GlobalLocalIndex = this->Index_Host.get();

	//get a device version of free-slip data
	STPFreeSlipData device_buffer(Data);
	device_buffer.GlobalLocalIndex = this->Index_Device.get();
	this->Data_Device = STPSmartDeviceMemory::makeDevice<STPFreeSlipData>();
	STPcudaCheckErr(cudaMemcpy(this->Data_Device.get(), &device_buffer, sizeof(STPFreeSlipData), cudaMemcpyHostToDevice));
}

const uvec2& STPFreeSlipGenerator::getDimension() const {
	return this->Data.Dimension;
}

const uvec2& STPFreeSlipGenerator::getFreeSlipChunk() const {
	return this->Data.FreeSlipChunk;
}

const uvec2& STPFreeSlipGenerator::getFreeSlipRange() const {
	return this->Data.FreeSlipRange;
}

template<typename T>
STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<T> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<T>& buffer) const {
	return STPFreeSlipManagerAdaptor(buffer, *this);
}

//Explicit instantiation
#define GET_MANAGER(TYPE) \
template STP_API STPFreeSlipGenerator::STPFreeSlipManagerAdaptor<TYPE> STPFreeSlipGenerator::operator()(STPFreeSlipTextureBuffer<TYPE>&) const

GET_MANAGER(float);
GET_MANAGER(Sample);