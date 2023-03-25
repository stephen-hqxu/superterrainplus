#include "STPBiomefieldGenerator.h"

#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
//Biome
#include "STPBiomeRegistry.h"

//GLM
#include <glm/gtc/type_ptr.hpp>

#include <cstring>

using namespace STPDemo;
using namespace SuperTerrainPlus;

using glm::uvec2;
using glm::vec2;
using glm::value_ptr;

using std::move;

STPBiomefieldGenerator::STPBiomefieldGenerator(const STPCommonCompiler& program, const uvec2 dimension, const unsigned int interpolation_radius)
	: STPDiversityGenerator(), MapSize(dimension), KernelProgram(program), InterpolationRadius(interpolation_radius), BufferPool(this->MapSize) {
	//init our device generator
	//our heightfield setting only available in OCEAN biome for now
	this->initGenerator();

	//create a CUDA memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.location.id = 0;
	this->HistogramCacheDevice = STPSmartDeviceObject::makeMemPool(pool_props);
	//it's pretty hard to predict
	constexpr size_t avg_bin_per_pixel = 2u, deg_para = 5u;
	cuuint64_t release_thres = this->MapSize.x * this->MapSize.y * (sizeof(unsigned int)
		+ sizeof(SuperTerrainPlus::STPAlgorithm::STPSingleHistogram::STPBin) * avg_bin_per_pixel) * deg_para;
	cudaMemPoolSetAttribute(this->HistogramCacheDevice.get(), cudaMemPoolAttrReleaseThreshold, &release_thres);
}

void STPBiomefieldGenerator::initGenerator() {
	//global pointers
	CUmodule program = this->KernelProgram.getProgram();
	CUdeviceptr biome_prop;
	size_t biome_propSize;
	//get names and start copying
	const auto& name = this->KernelProgram.getBiomefieldName();
	STP_CHECK_CUDA(cuModuleGetFunction(&this->GeneratorEntry, program, name.at("generateMultiBiomeHeightmap").c_str()));
	STP_CHECK_CUDA(cuModuleGetGlobal(&biome_prop, &biome_propSize, program, name.at("BiomeTable").c_str()));

	//copy biome properties
	//currently we have two biomes
	STPBiomeProperty* biomeTable_buffer;
	STP_CHECK_CUDA(cuMemAllocHost(reinterpret_cast<void**>(&biomeTable_buffer), biome_propSize));
	//copy to host buffer
	biomeTable_buffer[0] = static_cast<STPBiomeProperty>(STPBiomeRegistry::Ocean);
	biomeTable_buffer[1] = static_cast<STPBiomeProperty>(STPBiomeRegistry::Plains);
	//copy everything to device
	STP_CHECK_CUDA(cuMemcpyHtoD(biome_prop, biomeTable_buffer, biome_propSize));

	STP_CHECK_CUDA(cuMemFreeHost(biomeTable_buffer));
}

using namespace SuperTerrainPlus::STPAlgorithm;
using FiltExec = STPSingleHistogramFilter::STPFilterBuffer::STPExecutionType;

STPBiomefieldGenerator::STPHistogramBufferCreator::STPHistogramBufferCreator(const uvec2& mapDim) noexcept :
	BufferExecution(mapDim.x * mapDim.y < 100'000u ? FiltExec::Serial : FiltExec::Parallel) {

}

inline auto STPBiomefieldGenerator::STPHistogramBufferCreator::operator()() const {
	return STPSingleHistogramFilter::STPFilterBuffer(this->BufferExecution);
}
	
void STPBiomefieldGenerator::operator()(const STPNearestNeighbourFloatWTextureBuffer& heightmap_buffer,
	const STPNearestNeighbourSampleRTextureBuffer& biomemap_buffer, const vec2 offset) {
	//this function is called from multiple threads, consolidate the device context before calling any driver API function
	STP_CHECK_CUDA(cudaSetDevice(0));

	int Mingridsize, blocksize;
	//smart launch configuration
	STP_CHECK_CUDA(cuOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, this->GeneratorEntry, nullptr, 0u, 0));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		//under-sampled heightmap, and super-sample it back with interpolation
		Dimgridsize = (this->MapSize + Dimblocksize - 1u) / Dimblocksize;

	//retrieve raw texture
	const STPNearestNeighbourFloatWTextureBuffer::STPMergedBuffer heightmap_mem(
		heightmap_buffer, STPNearestNeighbourFloatWTextureBuffer::STPMemoryLocation::DeviceMemory);
	float* const heightmap = heightmap_mem.getDevice();
	//we only need host memory on biome map
	const STPNearestNeighbourSampleRTextureBuffer::STPMergedBuffer biomemap_mem(
		biomemap_buffer, STPNearestNeighbourSampleRTextureBuffer::STPMemoryLocation::HostMemory);
	const SuperTerrainPlus::STPDiversity::Sample* const biomemap = biomemap_mem.getHost();
	//get the stream, both buffer use the same stream
	const cudaStream_t stream = heightmap_buffer.DeviceMemInfo.second;

	//histogram filter
	STPSingleHistogramFilter::STPFilterBuffer histogram_buffer = this->BufferPool.requestObject();
	const STPSingleHistogram histogram_h = this->GenerateBiomeHistogram(
		biomemap, biomemap_buffer.NeighbourInfo, histogram_buffer, this->InterpolationRadius);

	/* ------------------------------------------ populate device memory ------------------------------------------- */
	//calculate the size of allocation, in number of element
	const auto [bin_size, offset_size] = histogram_buffer.size();
	const STPSmartDeviceMemory::STPStreamedDevice<STPSingleHistogram::STPBin> bin_device =
		STPSmartDeviceMemory::makeStreamedDevice<STPSingleHistogram::STPBin>(this->HistogramCacheDevice.get(), stream, bin_size);
	const STPSmartDeviceMemory::STPStreamedDevice<unsigned int> offset_device =
		STPSmartDeviceMemory::makeStreamedDevice<unsigned int>(this->HistogramCacheDevice.get(), stream, offset_size);
	
	const STPSingleHistogram histogram_d { bin_device.get(), offset_device.get() };
	//and copy, remember to use byte size
	//safe to cast away const because the memory assigned (see above) are non-const
	STP_CHECK_CUDA(cudaMemcpyAsync(const_cast<STPSingleHistogram::STPBin*>(histogram_d.Bin), histogram_h.Bin,
		bin_size * sizeof(STPSingleHistogram::STPBin), cudaMemcpyHostToDevice, stream));
	STP_CHECK_CUDA(cudaMemcpyAsync(const_cast<unsigned int*>(histogram_d.HistogramStartOffset), histogram_h.HistogramStartOffset,
		offset_size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
	
	//returning the buffer back to the pool, make sure all copies are done
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	this->BufferPool.returnObject(move(histogram_buffer));

	/* -------------------------------------- launch kernel ----------------------------------------- */
	const float2 gpu_offset = make_float2(offset.x, offset.y);
	constexpr static size_t BufferSize = sizeof(heightmap) + sizeof(histogram_d) + sizeof(gpu_offset);
	size_t buffer_size = BufferSize;
	unsigned char buffer[BufferSize];

	unsigned char* current_buffer = buffer;
	std::memcpy(current_buffer, &heightmap, sizeof(heightmap));
	current_buffer += sizeof(heightmap);
	std::memcpy(current_buffer, &histogram_d, sizeof(histogram_d));
	current_buffer += sizeof(histogram_d);
	std::memcpy(current_buffer, &gpu_offset, sizeof(gpu_offset));

	void* config[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &buffer_size,
		CU_LAUNCH_PARAM_END
	};
	STP_CHECK_CUDA(cuLaunchKernel(this->GeneratorEntry,
		Dimgridsize.x, Dimgridsize.y, 1u,
		Dimblocksize.x, Dimblocksize.y, 1u,
		0u, stream, nullptr, config));

	//free histogram device memory
	//STPBin is a POD-type so can be freed with no problem
	//histogram_d will be recycled after, the pointer will reside in the stream
}