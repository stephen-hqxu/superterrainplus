#include <SuperRealism+/Utility/STPAsyncAccelBuilder.h>

#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//OptiX
#include <optix.h>
#include <SuperRealism+/Utility/STPRendererErrorHandler.hpp>

using std::memory_order;

using namespace SuperTerrainPlus::STPRealism;

STPAsyncAccelBuilder::STPAsyncAccelBuilder() :
	AccelStruct{ }, FrontBuffer(AccelStruct), BackBuffer(AccelStruct + 1), AccelStatusFlag { false, false } {

}

bool STPAsyncAccelBuilder::build(OptixDeviceContext context, cudaStream_t stream, cudaMemPool_t memPool,
	const OptixAccelBuildOptions& accelOptions, const OptixBuildInput* buildInputs, unsigned int numBuildInputs,
	CUdeviceptr tempBuffer, size_t tempBufferSize, size_t outputBufferSize, const OptixAccelEmitDesc* emittedProperties,
	unsigned int numEmittedProperties) {
	if (accelOptions.operation == OPTIX_BUILD_OPERATION_UPDATE) {
		//In the future if we want to add support for update, since the update operation needs to be done in-place,
		//we don't need to allocate a memory and can do it directly on the front buffer.
		//The drawback is this operation needs to be synchronous,
		//because it would be too expensive to copy the front buffer content to the back buffer and relocate then update;
		//also we need to make sure front buffer is always available after this function returns.
		//But update is much faster than build so we are fine.
		throw STPException::STPUnsupportedFunctionality("The acceleration structure builder utility currently does not work with update operation");
	}
	if (this->AccelStatusFlag.IsBackBufferBusy.load(memory_order::memory_order_acquire)) {
		return false;
	}

	//allocate memory for back buffer
	auto& [back_mem, back_handle] = *this->BackBuffer;
	back_mem = STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(memPool, stream, outputBufferSize);
	//build
	this->AccelStatusFlag.IsBackBufferBusy.store(true, memory_order::memory_order_release);
	STP_CHECK_OPTIX(optixAccelBuild(context, stream, &accelOptions, buildInputs, numBuildInputs, tempBuffer,
		tempBufferSize, reinterpret_cast<CUdeviceptr>(back_mem.get()), outputBufferSize, &back_handle,
		emittedProperties, numEmittedProperties));

	//get a callback to notify us when build has finished
	//this is called from CUDA thread so we want to use atomic
	static auto notifyBuildComplete = [](void* data) -> void {
		STPBuildStatus& status = *reinterpret_cast<STPBuildStatus*>(data);
		status.IsBackBufferBusy.store(false, memory_order::memory_order_release);
		status.HasPendingSwap.store(true, memory_order::memory_order_release);
	};
	STP_CHECK_CUDA(cudaLaunchHostFunc(stream, notifyBuildComplete, &this->AccelStatusFlag));

	return true;
}

bool STPAsyncAccelBuilder::swapHandle() {
	if (this->AccelStatusFlag.IsBackBufferBusy.load(memory_order::memory_order_acquire)
		|| !this->AccelStatusFlag.HasPendingSwap.load(memory_order::memory_order_acquire)) {
		return false;
	}

	std::swap(const_cast<STPAccelStructBuffer*&>(this->FrontBuffer), this->BackBuffer);
	this->AccelStatusFlag.HasPendingSwap.store(false, memory_order::memory_order_release);
	//return back buffer back to its originated memory pool and reset everything
	auto& [back_mem, back_handle] = *this->BackBuffer;
	back_mem.reset();
	back_handle = 0ull;
	return true;
}

OptixTraversableHandle STPAsyncAccelBuilder::getTraversableHandle() const {
	return this->FrontBuffer->second;
}