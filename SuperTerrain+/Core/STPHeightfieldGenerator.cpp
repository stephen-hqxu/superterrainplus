#include <SuperTerrain+/World/Chunk/STPHeightfieldGenerator.h>

//Simulator
#include <SuperTerrain+/World/Chunk/FreeSlip/STPFreeSlipManager.cuh>
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidEnvironment.h>

#include <type_traits>
#include <memory>
//CUDA Kernel
#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

//Template definition for the smart device memory
#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>

using namespace SuperTerrainPlus::STPCompute;

using std::vector;
using std::mutex;
using std::unique_lock;
using std::optional;
using std::move;
using std::make_unique;

//GLM
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;


STPHeightfieldGenerator::STPHeightfieldGenerator(const STPEnvironment::STPChunkSetting& chunk_settings, const STPEnvironment::STPHeightfieldSetting& heightfield_settings,
	const STPDiversityGenerator& diversity_generator, unsigned int hint_level_of_concurrency)
	: generateHeightmap(diversity_generator), Heightfield_Setting_h(heightfield_settings), 
	FreeSlipTable(chunk_settings.FreeSlipChunk, chunk_settings.MapSize) {
	if (!chunk_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPChunkSetting are not validated");
	}
	if (!heightfield_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPHeightfieldSetting are not validated");
	}

	const unsigned int num_pixel = this->FreeSlipTable.getDimension().x * this->FreeSlipTable.getDimension().y,
		num_freeslip_pixel = this->FreeSlipTable.getFreeSlipRange().x * this->FreeSlipTable.getFreeSlipRange().y;
	this->TextureBufferAttr.TexturePixel = num_pixel;

	//allocating space
	//heightfield settings
	this->Heightfield_Setting_d = STPSmartDeviceMemory::makeDevice<STPEnvironment::STPHeightfieldSetting>();
	STPcudaCheckErr(cudaMemcpy(this->Heightfield_Setting_d.get(), &this->Heightfield_Setting_h, sizeof(STPEnvironment::STPHeightfieldSetting), cudaMemcpyHostToDevice));
	
	//create memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.location.id = 0;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	STPcudaCheckErr(cudaMemPoolCreate(&this->MapCacheDevice, &pool_props));
	//TODO: smartly determine the average memory pool size
	cuuint64_t release_thres = (sizeof(float) + sizeof(unsigned short)) * num_freeslip_pixel * hint_level_of_concurrency;
	STPcudaCheckErr(cudaMemPoolSetAttribute(this->MapCacheDevice, cudaMemPoolAttrReleaseThreshold, &release_thres));
	this->TextureBufferAttr.DeviceMemPool = this->MapCacheDevice;

	//Initialise random number generator maps
	this->RNG_Map = STPHeightfieldKernel::curandInit(this->Heightfield_Setting_h.Seed, this->Heightfield_Setting_h.RainDropCount);
}

STPHeightfieldGenerator::~STPHeightfieldGenerator() {
	STPcudaCheckErr(cudaMemPoolDestroy(this->MapCacheDevice));
	//device ptrs are deleted with custom deleter
}

void STPHeightfieldGenerator::operator()(STPMapStorage& args, STPGeneratorOperation operation) const {
	//check the availiability of the engine
	if (this->RNG_Map == nullptr) {
		return;
	}
	if (operation == 0u) {
		//no operation is specified, nothing can be done
		return;
	}

	//Retrieve all flags
	static auto isFlagged = [](STPGeneratorOperation op, STPGeneratorOperation flag) constexpr -> bool {
		return (op & flag) != 0u;
	};
	const bool flag[3] = {
		isFlagged(operation, STPHeightfieldGenerator::HeightmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Erosion),
		isFlagged(operation, STPHeightfieldGenerator::RenderingBufferGeneration)
	};
	STPcudaCheckErr(cudaSetDevice(0));

	//creating stream so cpu thread can calculate all chunks altogether
	//if exception is thrown during exception, stream will be the last object to be deleted, automatically
	optional<STPSmartStream> stream_buffer;
	cudaStream_t stream;
	//limit the scope for std::optional to control the destructor call
	{
		//heightmap
		optional<STPFreeSlipFloatTextureBuffer> heightmap_buffer;
		optional<STPFreeSlipRenderTextureBuffer> heightfield_buffer;
		optional<STPFreeSlipGenerator::STPFreeSlipFloatManagerAdaptor> heightmap_adaptor;
		//biomemap
		optional<STPFreeSlipSampleTextureBuffer> biomemap_buffer;
		optional<STPFreeSlipGenerator::STPFreeSlipSampleManagerAdaptor> biomemap_adaptor;

		//setup phase
		//we want the stream to not be blocked by default stream
		{
			unique_lock<mutex> stream_lock(this->StreamPool_lock);
			if (this->StreamPool.empty()) {
				//create a new stream
				stream_buffer.emplace(cudaStreamNonBlocking);
			}
			else {
				//grab an exisiting stream
				stream_buffer.emplace(move(this->StreamPool.front()));
				this->StreamPool.pop();
			}
		}
		stream = *stream_buffer;

		//Flag: HeightmapGeneration
		if (flag[0]) {
			//generate a new heightmap using diversity generator and store it to the output later
			//copy biome map to device, and allocate heightmap
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{ 1u, STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly, stream };
			STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData biomemap_data{ 1u, STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadOnly, stream };

			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
			biomemap_buffer.emplace(args.Biomemap, biomemap_data, this->TextureBufferAttr);
			biomemap_adaptor.emplace(this->FreeSlipTable(*biomemap_buffer));

			this->generateHeightmap(*heightmap_buffer, *biomemap_adaptor, args.HeightmapOffset, stream);
		}
		else {
			//no generation, use existing
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{ 1u, STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadWrite, stream };
			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
		}

		//Flag: Erosion
		if (flag[1]) {
			//prepare free-slip utility for heightmap
			heightmap_adaptor.emplace(this->FreeSlipTable(*heightmap_buffer));
			STPFreeSlipFloatManager heightmap_slip = (*heightmap_adaptor)(STPFreeSlipLocation::DeviceMemory);

			STPHeightfieldKernel::hydraulicErosion(heightmap_slip, this->Heightfield_Setting_d.get(),
				this->Heightfield_Setting_h.getErosionBrushSize(), this->Heightfield_Setting_h.RainDropCount, this->RNG_Map.get(), stream);
		}

		//Flag: RenderingBufferGeneration
		if (flag[2]) {
			//allocate formation memory
			STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData heightfield_rendering{ 1u, STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly, stream };
			heightfield_buffer.emplace(args.Heightfield16UI, heightfield_rendering, this->TextureBufferAttr);

			STPHeightfieldKernel::texture32Fto16(
				(*heightmap_buffer)(STPFreeSlipLocation::DeviceMemory), (*heightfield_buffer)(STPFreeSlipLocation::DeviceMemory),
				this->FreeSlipTable.getFreeSlipRange(), 1u, stream);
		}

		//Store the result accordingly
		//copy the result back to the host
		//it will call the destructor in texture buffer (optional calls it when goes out of scope), and result will be copied back using CUDA stream
		//this operation is stream ordered
	}

	//waiting for finish before release the stream back to the pool
	STPcudaCheckErr(cudaStreamSynchronize(stream));
	{
		unique_lock<mutex> stream_lock(this->StreamPool_lock);
		this->StreamPool.emplace(move(*stream_buffer));
	}
}