#include <SuperTerrain+/World/Chunk/STPHeightfieldGenerator.h>

//Simulator
#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <algorithm>

#include <glm/exponential.hpp>

using glm::uvec2;
using glm::vec2;

using std::vector;
using std::make_pair;

using namespace SuperTerrainPlus;

inline STPSmartDeviceObject::STPStream STPHeightfieldGenerator::STPStreamCreator::operator()() const {
	//we want the stream to not be blocked by default stream
	return STPSmartDeviceObject::makeStream(cudaStreamNonBlocking);
}

STPHeightfieldGenerator::STPRNGCreator::STPRNGCreator(const STPEnvironment::STPHeightfieldSetting& heightfield_setting) :
	Seed(heightfield_setting.Seed),
	Length(heightfield_setting.Erosion.RainDropCount) {

}

inline STPSmartDeviceMemory::STPDevice<STPHeightfieldGenerator::STPcurandRNG[]>
	STPHeightfieldGenerator::STPRNGCreator::operator()(const cudaStream_t stream) const {
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
static T generateErosionBrush(const unsigned int freeslip_rangeX, const unsigned int erosion_radius) {
	//radius must be greater than 0
	STP_ASSERTION_NUMERIC_DOMAIN(erosion_radius > 0u, "Erosion brush radius must be a positive integer");
	//none of the component should be zero
	STP_ASSERTION_NUMERIC_DOMAIN(freeslip_rangeX > 0u, "Free-slip range row count should be positive");
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

/**
 * @brief Create a nearest neighbour information.
 * @param map_dim The dimension of the map.
 * @param nn_chunk The number of nearest-neighbour of any given chunk.
 * @return The information.
*/
inline static STPNearestNeighbourInformation createNeighbourInfo(const uvec2 map_dim, const uvec2 nn_chunk) {
	return STPNearestNeighbourInformation {
		map_dim,
		nn_chunk,
		map_dim * nn_chunk
	};
}

STPHeightfieldGenerator::STPHeightfieldGenerator(const STPGeneratorSetup& setup) : STPHeightfieldGenerator(setup, *setup.ChunkSetting) {

}

STPHeightfieldGenerator::STPHeightfieldGenerator(const STPGeneratorSetup& setup, const STPEnvironment::STPChunkSetting& chk_setting) :
	HeightfieldSettingHost(*setup.HeightfieldSetting),
	NoNeighbour(createNeighbourInfo(chk_setting.MapSize, uvec2(1u))),
	DiversityNeighbour(createNeighbourInfo(chk_setting.MapSize, chk_setting.DiversityNearestNeighbour)),
	ErosionNeighbour(createNeighbourInfo(chk_setting.MapSize, chk_setting.ErosionNearestNeighbour)),
	generateHeightmap(*setup.DiversityGenerator),
	ErosionBrush(generateErosionBrush<STPErosionBrushMemory>(chk_setting.ErosionNearestNeighbour.x * chk_setting.MapSize.x,
		this->HeightfieldSettingHost.Erosion.ErosionBrushRadius)),
	RNGPool(this->HeightfieldSettingHost) {
	chk_setting.validate();
	this->HeightfieldSettingHost.validate();

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
	//TODO: smartly determine the average memory pool size, for example based on highest memory usage in a time period
	cuuint64_t release_thres = 104'857'600;//that's 100MB
	STP_CHECK_CUDA(cudaMemPoolSetAttribute(this->MapCacheDevice.get(), cudaMemPoolAttrReleaseThreshold, &release_thres));
}

#define PREPARE_GENERATION_DATA() STPSmartDeviceObject::STPStream smart_stream = this->StreamPool.requestObject(); \
const cudaStream_t stream = smart_stream.get(); \
const auto device_object = make_pair(this->MapCacheDevice.get(), stream)

#define CLEANUP_GENERATION_DATA() this->StreamPool.returnObject(move(smart_stream))

void STPHeightfieldGenerator::generate(float* const heightfield, const STPDiversity::Sample* const* const biomemap, const vec2 offset) {
	PREPARE_GENERATION_DATA();
	{
		//generate a new heightmap using diversity generator and store it to the output
		const STPNearestNeighbourFloatWTextureBuffer heightmap_buffer(&heightfield, this->NoNeighbour, device_object);
		const STPNearestNeighbourSampleRTextureBuffer samplemap_buffer(biomemap, this->DiversityNeighbour, device_object);

		this->generateHeightmap(heightmap_buffer, samplemap_buffer, offset);
	}
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	CLEANUP_GENERATION_DATA();
}

void STPHeightfieldGenerator::erode(float* const* const heightfield_original, unsigned short* const* const heightfield_low) {
	PREPARE_GENERATION_DATA();
	STPSmartDeviceMemory::STPDevice<STPcurandRNG[]> rng_buffer = move(this->RNGPool.requestObject(stream));
	//limit the scope of texture buffer to ensure their memory is sync'ed and freed at destruction before we return our memory back to the pool
	{
		const STPNearestNeighbourFloatRWTextureBuffer heightmap_buffer(heightfield_original, this->ErosionNeighbour, device_object);
		const STPNearestNeighbourRenderWTextureBuffer low_heightmap_buffer(heightfield_low, this->ErosionNeighbour, device_object);
		//create merged memory from the texture buffer
		STPNearestNeighbourFloatRWTextureBuffer::STPMergedBuffer heightmap_merged(heightmap_buffer,
			STPNearestNeighbourFloatRWTextureBuffer::STPMemoryLocation::DeviceMemory);
		STPNearestNeighbourRenderWTextureBuffer::STPMergedBuffer low_heightmap_merged(low_heightmap_buffer,
			STPNearestNeighbourRenderWTextureBuffer::STPMemoryLocation::DeviceMemory);

		//erosion
		STPHeightfieldKernel::hydraulicErosion(heightmap_merged.getDevice(), this->RainDropSettingDevice.get(),
			this->ErosionNeighbour, this->ErosionBrush.ErosionBrushRawData,
			this->HeightfieldSettingHost.Erosion.RainDropCount, rng_buffer.get(), stream);
		//generate low quality heightfield
		STPHeightfieldKernel::texture32Fto16(heightmap_merged.getDevice(), low_heightmap_merged.getDevice(),
			this->ErosionNeighbour.TotalMapSize, stream);

		//destruction of memory is streamed order, wait for the stream after those texture buffer die
	}
	STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	this->RNGPool.returnObject(move(rng_buffer));
	CLEANUP_GENERATION_DATA();
}