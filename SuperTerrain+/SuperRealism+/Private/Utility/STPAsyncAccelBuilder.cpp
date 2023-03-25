#include <SuperRealism+/Utility/STPAsyncAccelBuilder.h>

#include <SuperTerrain+/Exception/STPUnimplementedFeature.h>

//OptiX
#include <optix.h>
#include <SuperRealism+/Utility/STPRendererErrorHandler.hpp>

using namespace SuperTerrainPlus::STPRealism;

STPAsyncAccelBuilder::STPAsyncAccelBuilder() :
	AccelStruct{ }, FrontBuffer(AccelStruct), BackBuffer(AccelStruct + 1) {

}

OptixTraversableHandle STPAsyncAccelBuilder::build(const STPBuildInformation& buildInfo, const cudaMemPool_t memPool) {
	const auto [context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer, tempBufferSize,
		outputBufferSize, emittedProperties, numEmittedProperties] = buildInfo;

	if (accelOptions->operation == OPTIX_BUILD_OPERATION_UPDATE) {
		//In the future if we want to add support for update, since the update operation needs to be done in-place,
		//we don't need to allocate a memory and can do it directly on the front buffer.
		//The drawback is this operation needs to be synchronous,
		//because it would be too expensive to copy the front buffer content to the back buffer and relocate then update;
		//also we need to make sure front buffer is always available after this function returns.
		//But update is much faster than build so we are fine.
		throw STP_UNIMPLEMENTED_FEATURE_CREATE("The acceleration structure builder utility currently does not work with update operation");
	}
	//allocate memory for back buffer
	auto& [back_mem, back_handle] = *this->BackBuffer;
	back_mem = STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(memPool, stream, outputBufferSize);
	
	//build
	STP_CHECK_OPTIX(optixAccelBuild(context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
		tempBufferSize, reinterpret_cast<CUdeviceptr>(back_mem.get()), outputBufferSize, &back_handle,
		emittedProperties, numEmittedProperties));
	return back_handle;
}

OptixTraversableHandle STPAsyncAccelBuilder::compact(const STPCompactInformation& compactInfo, const cudaMemPool_t memPool) {
	const auto [context, stream, outputBufferSize] = compactInfo;

	auto& [back_mem, back_handle] = *this->BackBuffer;
	//allocate spare memory
	OptixTraversableHandle compactHandle;
	STPSmartDeviceMemory::STPStreamedDevice<unsigned char[]> compactMem =
		STPSmartDeviceMemory::makeStreamedDevice<unsigned char[]>(memPool, stream, outputBufferSize);

	//compact
	STP_CHECK_OPTIX(optixAccelCompact(context, stream, back_handle, reinterpret_cast<CUdeviceptr>(compactMem.get()), outputBufferSize, &compactHandle));
	//since everything happens in the same stream, the old back buffer will be freed after compaction is done
	back_mem = std::move(compactMem);
	back_handle = compactHandle;

	return back_handle;
}

void STPAsyncAccelBuilder::swapHandle() noexcept {
	std::swap(const_cast<STPAccelStructBuffer*&>(this->FrontBuffer), this->BackBuffer);
	//return back buffer back to its originated memory pool and reset everything
	*this->BackBuffer = STPAccelStructBuffer { };
}

OptixTraversableHandle STPAsyncAccelBuilder::getTraversableHandle() const noexcept {
	return this->FrontBuffer->second;
}