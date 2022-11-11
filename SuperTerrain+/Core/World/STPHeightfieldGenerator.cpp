#include <SuperTerrain+/World/Chunk/STPHeightfieldGenerator.h>

//Simulator
#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

#include <algorithm>
#include <cassert>

//GLM
#include <glm/exponential.hpp>

using std::vector;
using std::optional;
using std::move;

using namespace SuperTerrainPlus;

inline STPSmartDeviceObject::STPStream STPHeightfieldGenerator::STPStreamCreator::operator()() const {
	//we want the stream to not be blocked by default stream
	return STPSmartDeviceObject::makeStream(cudaStreamNonBlocking);
}

STPHeightfieldGenerator::STPRNGCreator::STPRNGCreator(const STPEnvironment::STPHeightfieldSetting& heightfield_setting) :
	Seed(heightfield_setting.Seed),
	Length(heightfield_setting.Erosion.RainDropCount) {

}

inline STPSmartDeviceMemory::STPDeviceMemory<STPHeightfieldGenerator::STPcurandRNG[]>
	STPHeightfieldGenerator::STPRNGCreator::operator()(cudaStream_t stream) const {
	return STPHeightfieldKernel::curandInit(this->Seed, this->Length, stream);
}

/**
 * @brief Generate the erosion brush.
 * @tparam T The erosion brush memory
 * @param freeslip_rangeX  The number of element on the free-slip heightmap in the free-slip chunk range in X direction,
 * i.e., the X dimension of the free-slip map.
 * @param erosion_radius The radius of erosion.
 * @return The memory containing generated erosion brush.
*/
template<class T>
static T generateErosionBrush(unsigned int freeslip_rangeX, unsigned int erosion_radius) {
	if (erosion_radius == 0u) {
		//radius must be greater than 0
		throw STPException::STPBadNumericRange("Erosion brush radius must be a positive integer");
	}
	if (freeslip_rangeX == 0u) {
		//none of the component should be zero
		throw STPException::STPBadNumericRange("Free-slip range row count should be positive");
	}
	/* -------------------------------------- Generate Erosion Brush ------------------------------- */
	const int radius = static_cast<int>(erosion_radius),
		radiusSqr = radius * radius;
	double weightSum = 0.0;
	//temporary cache for generation
	vector<int> indexCache;
	vector<float> weightCache;

	indexCache.reserve(radiusSqr);
	weightCache.reserve(radiusSqr);
	//calculate the brushing weight
	for (int brushY = -radius; brushY <= radius; brushY++) {
		for (int brushX = -radius; brushX <= radius; brushX++) {
			if (double sqrDst = static_cast<double>(brushX * brushX + brushY * brushY);
				sqrDst < radiusSqr) {
				//The brush lies within the erosion range
				const double currentbrushWeight = 1.0 - glm::sqrt(sqrDst) / radius;
				weightSum += currentbrushWeight;
				//store
				indexCache.emplace_back(brushY * static_cast<int>(freeslip_rangeX) + brushX);
				weightCache.emplace_back(static_cast<float>(currentbrushWeight));
			}
		}
	}
	//normalise the brush weight
	std::transform(weightCache.cbegin(), weightCache.cend(), weightCache.begin(),
		[weightFactor = 1.0 / weightSum](const float weight) { return static_cast<float>(weight * weightFactor); });

	assert(indexCache.size() == weightCache.size());
	/* ------------------------------------ Populate Device Memory ---------------------------------------------- */
	T memory;
	auto& [index_d, weight_d, brush_raw] = memory;
	index_d = STPSmartDeviceMemory::makeDevice<int[]>(indexCache.size());
	weight_d = STPSmartDeviceMemory::makeDevice<float[]>(weightCache.size());

	//copy
	STP_CHECK_CUDA(cudaMemcpy(index_d.get(), indexCache.data(),
		sizeof(int) * indexCache.size(), cudaMemcpyHostToDevice));
	STP_CHECK_CUDA(cudaMemcpy(weight_d.get(), weightCache.data(),
		sizeof(float) * weightCache.size(), cudaMemcpyHostToDevice));

	//store raw data
	brush_raw = STPErosionBrush {
		index_d.get(),
		weight_d.get(),
		static_cast<unsigned int>(indexCache.size())
	};

	return memory;
}

STPHeightfieldGenerator::STPHeightfieldGenerator(const STPGeneratorSetup& setup) :
	generateHeightmap(*setup.DiversityGenerator), HeightfieldSettingHost(*setup.HeightfieldSetting),
	ErosionBrush(generateErosionBrush<STPErosionBrushMemory>(setup.ChunkSetting->FreeSlipChunk.x * setup.ChunkSetting->MapSize.x,
		this->HeightfieldSettingHost.Erosion.ErosionBrushRadius)),
	RNGPool(this->HeightfieldSettingHost) {
	const STPEnvironment::STPChunkSetting& chunk_setting = *setup.ChunkSetting;

	chunk_setting.validate();
	this->HeightfieldSettingHost.validate();
	STPFreeSlipInformation& info = this->TextureBufferAttr.TextureInfo;
	info.Dimension = chunk_setting.MapSize;
	info.FreeSlipChunk = chunk_setting.FreeSlipChunk;
	info.FreeSlipRange = info.Dimension * info.FreeSlipChunk;

	//allocating space
	//heightfield settings
	this->RainDropSettingDevice = STPSmartDeviceMemory::makeDevice<STPEnvironment::STPRainDropSetting>();
	STP_CHECK_CUDA(cudaMemcpy(this->RainDropSettingDevice.get(), &this->HeightfieldSettingHost.Erosion,
		sizeof(STPEnvironment::STPRainDropSetting), cudaMemcpyHostToDevice));

	//create memory pool
	cudaMemPoolProps pool_props = { };
	pool_props.allocType = cudaMemAllocationTypePinned;
	pool_props.location.id = 0;
	pool_props.location.type = cudaMemLocationTypeDevice;
	pool_props.handleTypes = cudaMemHandleTypeNone;
	this->MapCacheDevice = STPSmartDeviceObject::makeMemPool(pool_props);
	//TODO: smartly determine the average memory pool size
	cuuint64_t release_thres = (sizeof(float) + sizeof(unsigned short)) * info.FreeSlipRange.x * info.FreeSlipRange.y
		* setup.ConcurrencyLevelHint;
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
	STPSmartDeviceMemory::STPDeviceMemory<STPcurandRNG[]> rng_buffer;
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
			rng_buffer = move(this->RNGPool.requestObject(stream));
			STPHeightfieldKernel::hydraulicErosion(
				(*heightmap_buffer)(STPFreeSlipFloatTextureBuffer::STPFreeSlipLocation::DeviceMemory),
				this->RainDropSettingDevice.get(), this->TextureBufferAttr.TextureInfo, this->ErosionBrush.ErosionBrushRawData,
				this->HeightfieldSettingHost.Erosion.RainDropCount, rng_buffer.get(), stream);
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
	if (rng_buffer) {
		//if we have previously grabbed a RNG from the pool, return it
		this->RNGPool.returnObject(move(rng_buffer));
	}
	this->StreamPool.returnObject(move(smart_stream));
}