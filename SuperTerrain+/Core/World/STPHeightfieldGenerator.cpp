#include <SuperTerrain+/World/Chunk/STPHeightfieldGenerator.h>

//Simulator
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

#include <type_traits>
#include <memory>
//CUDA Kernel
#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

using namespace SuperTerrainPlus;

using std::vector;
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

inline STPSmartDeviceObject::STPStream STPHeightfieldGenerator::STPStreamCreator::operator()() const {
	//we want the stream to not be blocked by default stream
	return STPSmartDeviceObject::makeStream(cudaStreamNonBlocking);
}

STPHeightfieldGenerator::STPRNGCreator::STPRNGCreator(const STPEnvironment::STPHeightfieldSetting& heightfield_setting) :
	Seed(heightfield_setting.Seed),
	Length(heightfield_setting.RainDropCount) {

}

inline STPSmartDeviceMemory::STPDeviceMemory<STPHeightfieldGenerator::STPcurandRNG[]>
	STPHeightfieldGenerator::STPRNGCreator::operator()(cudaStream_t stream) const {
	return STPHeightfieldKernel::curandInit(this->Seed, this->Length, stream);
}

STPHeightfieldGenerator::STPHeightfieldGenerator(const STPEnvironment::STPChunkSetting& chunk_settings,
	const STPEnvironment::STPHeightfieldSetting& heightfield_settings, const STPDiversityGenerator& diversity_generator,
	unsigned int hint_level_of_concurrency) :
	generateHeightmap(diversity_generator), Heightfield_Setting_h(heightfield_settings),
	ErosionBrush(chunk_settings.FreeSlipChunk.x * chunk_settings.MapSize.x, this->Heightfield_Setting_h.ErosionBrushRadius),
	RNGPool(this->Heightfield_Setting_h) {
	if (!chunk_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPChunkSetting are not validated");
	}
	if (!heightfield_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPHeightfieldSetting are not validated");
	}
	STPFreeSlipInformation& info = this->TextureBufferAttr.TextureInfo;
	info.Dimension = chunk_settings.MapSize;
	info.FreeSlipChunk = chunk_settings.FreeSlipChunk;
	info.FreeSlipRange = info.Dimension * info.FreeSlipChunk;

	//allocating space
	//heightfield settings
	this->Heightfield_Setting_d = STPSmartDeviceMemory::makeDevice<STPEnvironment::STPHeightfieldSetting>();
	STP_CHECK_CUDA(cudaMemcpy(this->Heightfield_Setting_d.get(), &this->Heightfield_Setting_h,
		sizeof(STPEnvironment::STPHeightfieldSetting), cudaMemcpyHostToDevice));

	//create memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.location.id = 0;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	this->MapCacheDevice = STPSmartDeviceObject::makeMemPool(pool_props);
	//TODO: smartly determine the average memory pool size
	cuuint64_t release_thres = (sizeof(float) + sizeof(unsigned short)) * info.FreeSlipRange.x * info.FreeSlipRange.y
		* hint_level_of_concurrency;
	STP_CHECK_CUDA(cudaMemPoolSetAttribute(this->MapCacheDevice.get(), cudaMemPoolAttrReleaseThreshold, &release_thres));
	this->TextureBufferAttr.DeviceMemPool = this->MapCacheDevice.get();
}

void STPHeightfieldGenerator::operator()(STPMapStorage& args, STPGeneratorOperation operation) const {
	if (operation == 0u) {
		//no operation is specified, nothing can be done
		return;
	}
	//Retrieve all flags
	static constexpr auto isFlagged = [](STPGeneratorOperation op, STPGeneratorOperation flag) constexpr -> bool {
		return (op & flag) != 0u;
	};
	const bool flag[3] = {
		isFlagged(operation, STPHeightfieldGenerator::HeightmapGeneration),
		isFlagged(operation, STPHeightfieldGenerator::Erosion),
		isFlagged(operation, STPHeightfieldGenerator::RenderingBufferGeneration)
	};
	STP_CHECK_CUDA(cudaSetDevice(0));

	//creating stream so CPU thread can calculate all chunks altogether
	//if exception is thrown during exception, stream will be the last object to be deleted, automatically
	STPSmartDeviceObject::STPStream smart_stream = this->StreamPool.requestObject();
	cudaStream_t stream = smart_stream.get();
	optional<STPSmartDeviceMemory::STPDeviceMemory<STPcurandRNG[]>> rng_buffer;
	//limit the scope for std::optional to control the destructor call
	{
		//heightmap
		optional<STPFreeSlipFloatTextureBuffer> heightmap_buffer;
		optional<STPFreeSlipRenderTextureBuffer> heightfield_buffer;
		//biomemap
		optional<STPFreeSlipSampleTextureBuffer> biomemap_buffer;

		//Flag: HeightmapGeneration
		if (flag[0]) {
			//generate a new heightmap using diversity generator and store it to the output later
			//copy biome map to device, and allocate heightmap
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{
				STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly,
				stream
			};
			STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData biomemap_data{
				STPFreeSlipSampleTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadOnly,
				stream
			};

			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
			biomemap_buffer.emplace(args.Biomemap, biomemap_data, this->TextureBufferAttr);

			this->generateHeightmap(*heightmap_buffer, *biomemap_buffer, this->TextureBufferAttr.TextureInfo, args.HeightmapOffset, stream);
		} else {
			//no generation, use existing
			STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData heightmap_data{
				STPFreeSlipFloatTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::ReadWrite,
				stream
			};
			heightmap_buffer.emplace(args.Heightmap32F, heightmap_data, this->TextureBufferAttr);
		}

		//Flag: Erosion
		if (flag[1]) {
			rng_buffer.emplace(this->RNGPool.requestObject(stream));
			STPHeightfieldKernel::hydraulicErosion(
				(*heightmap_buffer)(STPFreeSlipFloatTextureBuffer::STPFreeSlipLocation::DeviceMemory),
				this->Heightfield_Setting_d.get(), this->TextureBufferAttr.TextureInfo, this->ErosionBrush.getBrush(),
				this->Heightfield_Setting_h.RainDropCount, rng_buffer->get(), stream);
		}

		//Flag: RenderingBufferGeneration
		if (flag[2]) {
			//allocate formation memory
			STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData heightfield_rendering{
				STPFreeSlipRenderTextureBuffer::STPFreeSlipTextureData::STPMemoryMode::WriteOnly,
				stream
			};
			heightfield_buffer.emplace(args.Heightfield16UI, heightfield_rendering, this->TextureBufferAttr);

			STPHeightfieldKernel::texture32Fto16(
				(*heightmap_buffer)(STPFreeSlipFloatTextureBuffer::STPFreeSlipLocation::DeviceMemory), 
				(*heightfield_buffer)(STPFreeSlipRenderTextureBuffer::STPFreeSlipLocation::DeviceMemory),
				this->TextureBufferAttr.TextureInfo.FreeSlipRange, 1u, stream);
		}

		//Store the result accordingly
		//copy the result back to the host
		//it will call the destructor in texture buffer (optional calls it when goes out of scope), and result will be copied back using CUDA stream
		//this operation is stream ordered
	}

	//waiting for finish before release the data back to the pool
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	if (rng_buffer.has_value()) {
		//if we have previously grabbed a RNG from the pool, return it
		this->RNGPool.returnObject(move(*rng_buffer));
	}
	this->StreamPool.returnObject(move(smart_stream));
}