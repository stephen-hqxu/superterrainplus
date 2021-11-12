#include "STPBiomefieldGenerator.h"
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
//Biome
#include "STPBiomeRegistry.h"

//GLM
#include <glm/gtc/type_ptr.hpp>

using namespace STPDemo;
using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;
using glm::vec2;
using glm::value_ptr;

using std::unique_lock;
using std::move;
using std::mutex;

STPBiomefieldGenerator::STPBiomefieldGenerator(const STPCommonCompiler& program, STPSimplexNoiseSetting& simplex_setting, uvec2 dimension, unsigned int interpolation_radius)
	: STPDiversityGenerator(), KernelProgram(program), Noise_Setting(simplex_setting), MapSize(dimension), Simplex_Permutation(this->Noise_Setting), InterpolationRadius(interpolation_radius) {
	//init our device generator
	//our heightfield setting only available in OCEAN biome for now
	this->initGenerator();

	//create a cuda memory pool
	CUmemPoolProps pool_props = { };
	pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
	pool_props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
	pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	pool_props.location.id = 0;
	STPcudaCheckErr(cuMemPoolCreate(&this->HistogramCacheDevice, &pool_props));
	//it's pretty hard to predict
	constexpr size_t avg_bin_per_pixel = 2ull, deg_para = 5ull;
	cuuint64_t release_thres = this->MapSize.x * this->MapSize.y * (sizeof(unsigned int) + sizeof(SuperTerrainPlus::STPCompute::STPSingleHistogram::STPBin) * avg_bin_per_pixel) * deg_para;
	cuMemPoolSetAttribute(this->HistogramCacheDevice, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &release_thres);
}

STPBiomefieldGenerator::~STPBiomefieldGenerator() {
	STPcudaCheckErr(cuMemPoolDestroy(this->HistogramCacheDevice));
}

void STPBiomefieldGenerator::initGenerator() {
	//global pointers
	CUmodule program = this->KernelProgram.getProgram();
	CUdeviceptr biome_prop, dimension, half_dimension, permutation;
	size_t biome_propSize, dimensionSize, half_dimensionSize, permutationSize;
	//get names and start copying
	const auto& name = this->KernelProgram.getLoweredNameDictionary("STPMultiHeightGenerator");
	STPcudaCheckErr(cuModuleGetFunction(&this->GeneratorEntry, program, name.at("generateMultiBiomeHeightmap")));
	STPcudaCheckErr(cuModuleGetGlobal(&biome_prop, &biome_propSize, program, name.at("BiomeTable")));
	STPcudaCheckErr(cuModuleGetGlobal(&dimension, &dimensionSize, program, name.at("Dimension")));
	STPcudaCheckErr(cuModuleGetGlobal(&half_dimension, &half_dimensionSize, program, name.at("HalfDimension")));
	STPcudaCheckErr(cuModuleGetGlobal(&permutation, &permutationSize, program, name.at("Permutation")));
	//copy variables
	const vec2 halfSize = static_cast<vec2>(this->MapSize) * 0.5f;
	STPcudaCheckErr(cuMemcpyHtoD(dimension, value_ptr(this->MapSize), dimensionSize));
	STPcudaCheckErr(cuMemcpyHtoD(half_dimension, value_ptr(halfSize), half_dimensionSize));
	//note that we are copying permutation to device, the underlying pointers are managed by this class
	STPcudaCheckErr(cuMemcpyHtoD(permutation, &this->Simplex_Permutation(), permutationSize));

	//copy biome properties
	//currently we have two biomes
	STPBiomeProperty* biomeTable_buffer;
	STPcudaCheckErr(cuMemAllocHost(reinterpret_cast<void**>(&biomeTable_buffer), biome_propSize));
	constexpr size_t one_biomeprop_size = sizeof(STPBiomeProperty);
	//copy to host buffer
	memcpy(biomeTable_buffer, dynamic_cast<const STPBiomeProperty*>(&STPBiomeRegistry::OCEAN.getProperties()), one_biomeprop_size);
	memcpy(biomeTable_buffer + 1, dynamic_cast<const STPBiomeProperty*>(&STPBiomeRegistry::PLAINS.getProperties()), one_biomeprop_size);
	//copy everything to device
	STPcudaCheckErr(cuMemcpyHtoD(biome_prop, biomeTable_buffer, biome_propSize));

	STPcudaCheckErr(cuMemFreeHost(biomeTable_buffer));
}

//Memory Pool
#include <SuperTerrain+/Utility/Memory/STPMemoryPool.h>

using namespace SuperTerrainPlus::STPCompute;

//As a reminder, this is thread-safe
static SuperTerrainPlus::STPRegularMemoryPool CallbackDataPool;

struct STPBiomefieldGenerator::STPBufferReleaseData {
public:

	//The pool that the buffer will be returned to
	STPBiomefieldGenerator::STPHistogramBufferPool* Pool;
	//The buffer to be returned
	STPSingleHistogramFilter::STPHistogramBuffer_t Buffer;
	//The pool lock
	mutex* Lock;

};

void STPBiomefieldGenerator::returnBuffer(void* buffer_data) {
	STPBufferReleaseData* data = reinterpret_cast<STPBufferReleaseData*>(buffer_data);

	//return buffer to the buffer pool safely
	{
		unique_lock<mutex> buffer_lock(*data->Lock);
		data->Pool->emplace(move(data->Buffer));
	}

	if constexpr (!std::is_trivially_destructible_v<STPBufferReleaseData>) {
		data->~STPBufferReleaseData();
	}
	//free the data
	CallbackDataPool.release(buffer_data);
}
	
void STPBiomefieldGenerator::operator()(STPFreeSlipFloatTextureBuffer& heightmap_buffer, const STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor& biomemap_adaptor, vec2 offset, cudaStream_t stream) const {
	int Mingridsize, blocksize;
	//smart launch config
	STPcudaCheckErr(cuOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, this->GeneratorEntry, nullptr, 0ull, 0));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		//under-sampled heightmap, and super-sample it back with interpolation
		Dimgridsize = (this->MapSize + Dimblocksize - 1u) / Dimblocksize;

	//retrieve raw texture
	float* heightmap = heightmap_buffer(STPFreeSlipLocation::DeviceMemory);
	//we only need host memory on biome map
	STPFreeSlipSampleManager biomemap_manager = biomemap_adaptor(STPFreeSlipLocation::HostMemory);

	//histogram filter
	STPSingleHistogramFilter::STPHistogramBuffer_t histogram_buffer;
	{
		unique_lock<mutex> buffer_lock(this->BufferPoolLock);
		//try to grab a buffer
		if (this->BufferPool.empty()) {
			//no more buffer avilable? create a new one
			histogram_buffer = move(STPSingleHistogramFilter::createHistogramBuffer());
		}
		else {
			//otherwise grab an exisiting one
			histogram_buffer = move(this->BufferPool.front());
			this->BufferPool.pop();
		}
	}
	//start execution
	//host to host memory copy is always synchornous, so the host memory should be available now
	STPSingleHistogram histogram_h;
	{
		unique_lock<mutex> filter_lock(this->HistogramFilterLock);
		histogram_h = this->biome_histogram(biomemap_manager, histogram_buffer, this->InterpolationRadius);
	}
	//copy histogram to device
	STPSingleHistogram histogram_d;
	//calculate the size of allocation
	const uvec2& biome_dimension = biomemap_manager.Data->Dimension;
	const unsigned int num_pixel_biomemap = biome_dimension.x * biome_dimension.y;
	const size_t bin_size = histogram_h.HistogramStartOffset[num_pixel_biomemap] * sizeof(STPSingleHistogram::STPBin),
		offset_size = (num_pixel_biomemap + 1u) * sizeof(unsigned int);
	//the number of bin is the last element in the offset array
	STPcudaCheckErr(cudaMallocFromPoolAsync(&histogram_d.Bin, bin_size, this->HistogramCacheDevice, stream));
	STPcudaCheckErr(cudaMallocFromPoolAsync(&histogram_d.HistogramStartOffset, offset_size, this->HistogramCacheDevice, stream));
	//and copy
	STPcudaCheckErr(cudaMemcpyAsync(histogram_d.Bin, histogram_h.Bin, bin_size, cudaMemcpyHostToDevice, stream));
	STPcudaCheckErr(cudaMemcpyAsync(histogram_d.HistogramStartOffset, histogram_h.HistogramStartOffset, offset_size, cudaMemcpyHostToDevice, stream));

	//returning the buffer requires a stream callback
	//do a placement new call to construct since STPBufferReleaseData is not trivial
	STPBufferReleaseData* release_data = new(CallbackDataPool.request(sizeof(STPBufferReleaseData))) STPBufferReleaseData();
	release_data->Buffer = move(histogram_buffer);
	release_data->Lock = &this->BufferPoolLock;
	release_data->Pool = &this->BufferPool;
	STPcudaCheckErr(cuLaunchHostFunc(stream, &STPBiomefieldGenerator::returnBuffer, release_data));

	//launch kernel
	float2 gpu_offset = make_float2(offset.x, offset.y);
	size_t bufferSize = 32ull;
	unsigned char buffer[32];

	unsigned char* current_buffer = buffer;
	memcpy(current_buffer, &heightmap, sizeof(heightmap));
	current_buffer += sizeof(heightmap);
	memcpy(current_buffer, &histogram_d, sizeof(histogram_d));
	current_buffer += sizeof(histogram_d);
	memcpy(current_buffer, &gpu_offset, sizeof(gpu_offset));

	void* config[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &bufferSize,
		CU_LAUNCH_PARAM_END
	};
	STPcudaCheckErr(cuLaunchKernel(this->GeneratorEntry,
		Dimgridsize.x, Dimgridsize.y, 1u,
		Dimblocksize.x, Dimblocksize.y, 1u,
		0u, stream, nullptr, config));
	STPcudaCheckErr(cudaGetLastError());

	//free histogram device memory
	//STPBin is a POD-type so can be freed with no problem
	//histogram_d will be recycled after, the pointer will reside in the stream
	STPcudaCheckErr(cudaFreeAsync(histogram_d.Bin, stream));
	STPcudaCheckErr(cudaFreeAsync(histogram_d.HistogramStartOffset, stream));
}