#include "STPBiomefieldGenerator.h"
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
//Biome
#include "STPBiomeRegistry.h"

//GLM
#include <glm/gtc/type_ptr.hpp>

using namespace STPDemo;
using namespace SuperTerrainPlus;

using glm::uvec2;
using glm::vec2;
using glm::value_ptr;

using std::move;

STPBiomefieldGenerator::STPBiomefieldGenerator(const STPCommonCompiler& program, uvec2 dimension, unsigned int interpolation_radius)
	: STPDiversityGenerator(), MapSize(dimension), KernelProgram(program), InterpolationRadius(interpolation_radius) {
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

inline auto STPBiomefieldGenerator::STPHistogramBufferCreator::operator()() const {
	return STPSingleHistogramFilter::createHistogramBuffer();
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
	const Sample* const biomemap = biomemap_mem.getHost();
	//get the stream, both buffer use the same stream
	const cudaStream_t stream = heightmap_buffer.DeviceMemInfo.second;

	//histogram filter
	STPSingleHistogramFilter::STPHistogramBuffer_t histogram_buffer = this->BufferPool.requestObject();
	//start execution
	//host to host memory copy is always synchronous, so the host memory should be available now
	STPSingleHistogram histogram_h;
	//need to use biomemap neighbour information because we are generating histogram based on biome neighbours
	const STPNearestNeighbourInformation& biome_neighbour_info = biomemap_buffer.NeighbourInfo;

	histogram_h = this->biome_histogram(biomemap, biome_neighbour_info, histogram_buffer, this->InterpolationRadius);
	//copy histogram to device
	STPSingleHistogram histogram_d;
	//calculate the size of allocation
	const uvec2& biome_dimension = biome_neighbour_info.MapSize;
	const unsigned int num_pixel_biomemap = biome_dimension.x * biome_dimension.y;
	const size_t bin_size = histogram_h.HistogramStartOffset[num_pixel_biomemap] * sizeof(STPSingleHistogram::STPBin),
		offset_size = (num_pixel_biomemap + 1u) * sizeof(unsigned int);
	//the number of bin is the last element in the offset array
	STP_CHECK_CUDA(cudaMallocFromPoolAsync(&histogram_d.Bin, bin_size, this->HistogramCacheDevice.get(), stream));
	STP_CHECK_CUDA(cudaMallocFromPoolAsync(&histogram_d.HistogramStartOffset, offset_size, this->HistogramCacheDevice.get(), stream));
	//and copy
	STP_CHECK_CUDA(cudaMemcpyAsync(histogram_d.Bin, histogram_h.Bin, bin_size, cudaMemcpyHostToDevice, stream));
	STP_CHECK_CUDA(cudaMemcpyAsync(histogram_d.HistogramStartOffset, histogram_h.HistogramStartOffset, offset_size,
		cudaMemcpyHostToDevice, stream));
	//returning the buffer back to the pool, make sure all copies are done
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	this->BufferPool.returnObject(move(histogram_buffer));

	//launch kernel
	const float2 gpu_offset = make_float2(offset.x, offset.y);
	constexpr static size_t BufferSize = sizeof(heightmap) + sizeof(histogram_d) + sizeof(gpu_offset);
	size_t buffer_size = BufferSize;
	unsigned char buffer[BufferSize];

	unsigned char* current_buffer = buffer;
	memcpy(current_buffer, &heightmap, sizeof(heightmap));
	current_buffer += sizeof(heightmap);
	memcpy(current_buffer, &histogram_d, sizeof(histogram_d));
	current_buffer += sizeof(histogram_d);
	memcpy(current_buffer, &gpu_offset, sizeof(gpu_offset));

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
	STP_CHECK_CUDA(cudaFreeAsync(histogram_d.Bin, stream));
	STP_CHECK_CUDA(cudaFreeAsync(histogram_d.HistogramStartOffset, stream));
}