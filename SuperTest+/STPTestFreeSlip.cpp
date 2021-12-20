//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/GPGPU/FreeSlip
#include <SuperTerrain+/World/Chunk/FreeSlip/STPFreeSlipGenerator.h>

#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//Shared Test Data
#include "STPTestInformation.h"

#include <glm/gtc/type_ptr.hpp>

#include <type_traits>
#include <algorithm>
#include <optional>
#include <random>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;

using STPDiversity::Sample;

using glm::uvec2;

using std::vector;
using std::optional;
using std::unique_ptr;
using std::make_unique;
using std::make_pair;

template<typename T>
class FreeSlipBufferTester {
private:

	T* CurrentMergedTexture;
	//only for device memory location
	T* MergedDevice;
	unique_ptr<T[]> MergedHost;

	STPFreeSlipLocation CurrentLocation;

protected:

	constexpr static uvec2 SmallDimension = uvec2(4u);
	constexpr static uvec2 SmallChunkUnit = uvec2(2u);
	constexpr static unsigned int SmallSize = SmallDimension.x * SmallDimension.y * SmallChunkUnit.x * SmallChunkUnit.y;

	using CurrentFreeSlipBuffer = STPFreeSlipTextureBuffer<T>;
	using TestData = typename CurrentFreeSlipBuffer::STPFreeSlipTextureData;
	using TestMemoryMode = typename TestData::STPMemoryMode;

	T Reference[SmallSize];
	T Texture[SmallSize];
	vector<T*> TextureBuffer;

	inline static STPFreeSlipTextureAttribute SmallAttribute{ SmallSize, STPPinnedMemoryPool(), 0 };

	static T getRandomData() {
		static std::mt19937_64 Generator(GENERATE(take(1, random(0ull, 6666666666ull))));

		T value;
		if constexpr (std::is_floating_point_v<T>) {
			static std::uniform_real_distribution<T> Distribution(-6666.66f, 6666.66f);
			//2 d.p.
			value = roundf(Distribution(Generator) * 100.0f) / 100.0f;
		}
		else {
			static std::uniform_int_distribution<T> Distribution(0u, 16666u);
			value = Distribution(Generator);
		}
		return value;
	}

	void registerMergedBuffer(STPFreeSlipLocation location, T* buffer) {
		this->CurrentMergedTexture = buffer;
		this->CurrentLocation = location;

		if (this->CurrentLocation == STPFreeSlipLocation::DeviceMemory) {
			typedef FreeSlipBufferTester<T> FBT;
			//if it's a device memory, we need to copy it to host before we can use it
			this->MergedDevice = this->CurrentMergedTexture;
			this->MergedHost = make_unique<T[]>(FBT::SmallSize);
			STPcudaCheckErr(cudaMemcpyAsync(this->MergedHost.get(), this->MergedDevice, FBT::SmallSize * sizeof(T), cudaMemcpyDeviceToHost, 0));
			STPcudaCheckErr(cudaStreamSynchronize(0));

			//reassign pointer so later assertions can use it on host side
			this->CurrentMergedTexture = MergedHost.get();
		}
	}

	inline void compareHostDeviceBuffer(T* host_cache, TestMemoryMode mode) {
		REQUIRE_FALSE(host_cache == this->MergedDevice);
		CHECKED_ELSE(mode == TestMemoryMode::WriteOnly) {
			//we can't gurantee the data on both ends is the same on write only mode
			REQUIRE(std::equal(host_cache, host_cache + FreeSlipBufferTester<T>::SmallSize, this->MergedHost.get()));
		}
	}

	inline T* getBuffer() {
		return this->CurrentMergedTexture;
	}

	inline void updateDeviceBuffer() {
		if (this->CurrentLocation == STPFreeSlipLocation::DeviceMemory) {
			STPcudaCheckErr(cudaMemcpyAsync(this->MergedDevice, this->CurrentMergedTexture, FreeSlipBufferTester<T>::SmallSize * sizeof(T), cudaMemcpyHostToDevice, 0));
		}
		//there is no device buffer in host memory mode
	}

public:

	FreeSlipBufferTester() {
		//fill the reference texture with random value
		std::fill_n(this->Reference, FreeSlipBufferTester<T>::SmallSize, FreeSlipBufferTester<T>::getRandomData());
		
		//texture is the buffer that we are testing
		std::copy_n(this->Reference, FreeSlipBufferTester<T>::SmallSize, this->Texture);

		this->TextureBuffer.emplace_back(this->Texture);

		static bool AttrInit = false;
		if (!AttrInit) {
			FreeSlipBufferTester::SmallAttribute.DeviceMemPool = STPTestInformation::TestDeviceMemoryPool;
			AttrInit = true;
		}
	}

};

#define WRITE_MERGED() \
std::fill_n(OperatedMergedBuffer, CurrentTester::SmallSize, RandomData); \
this->updateDeviceBuffer(); \
REQUIRE_NOTHROW(TestBuffer.reset())

#define MERGED_AVAILABLE() REQUIRE(OperatedMergedBuffer[index] == this->Texture[index])
#define TEXTURE_WRITTEN() REQUIRE(this->Texture[index] == RandomData)

TEMPLATE_TEST_CASE_METHOD(FreeSlipBufferTester, "STPFreeSlipTextureBuffer can merge and disintegrate per-chunk texture following the memory mode", 
	"[GPGPU][FreeSlip][STPFreeSlipTextureBuffer]", float, Sample) {
	using CurrentTester = FreeSlipBufferTester<TestType>;

	WHEN("Some wrong numbers are provided to the texture buffer") {
		vector<TestType*> EmptyBuffer;
		STPFreeSlipTextureAttribute ZeroPixel = { 0u, { }, 0 };

		THEN("Creation of texture buffer should be prevented") {
			//no texture buffer
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(EmptyBuffer, { 1u }, CurrentTester::SmallAttribute), STPException::STPInvalidArgument);
			//no pixel
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(this->TextureBuffer, { 1u }, ZeroPixel), STPException::STPBadNumericRange);
			//no channel
			REQUIRE_THROWS_AS(CurrentFreeSlipBuffer(this->TextureBuffer, { }, CurrentTester::SmallAttribute), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A valid array of texture buffer and appropriate memory operation mode") {
		const TestMemoryMode Mode = GENERATE(values({
			TestMemoryMode::ReadOnly,
			TestMemoryMode::WriteOnly,
			TestMemoryMode::ReadWrite
		}));
		optional<CurrentFreeSlipBuffer> TestBuffer;

		REQUIRE_NOTHROW([&TestBuffer, Mode, this]() {
			TestData Data = { 1u, Mode, 0 };
			TestBuffer.emplace(this->TextureBuffer, Data, CurrentTester::SmallAttribute);
		}());

		WHEN("Texture buffer is unmerged") {

			THEN("Merge location is not available") {
				REQUIRE_THROWS_AS(TestBuffer->where() == STPFreeSlipLocation::HostMemory, STPException::STPMemoryError);
			}

		}

		WHEN("Merge the texture with said memory mode") {
			const STPFreeSlipLocation ChosenLocation = GENERATE(values({
				STPFreeSlipLocation::HostMemory,
				STPFreeSlipLocation::DeviceMemory
			}));
			//When using device merged buffer, we need to copy it back to host to be able to test it

			THEN("Merging should be successful and return a merged texture") {
				TestType* RawMergedBuffer;
				REQUIRE_NOTHROW([&RawMergedBuffer, &TestBuffer, ChosenLocation]() {
					RawMergedBuffer = (*TestBuffer)(ChosenLocation);
				}());
				this->registerMergedBuffer(ChosenLocation, RawMergedBuffer);

				AND_THEN("Merge location is correct and no reallocation when the merging operation is called again, instead the previously returned memory should be returned") {
					REQUIRE(TestBuffer->where() == ChosenLocation);
					REQUIRE((*TestBuffer)(ChosenLocation) == RawMergedBuffer);

					CHECKED_IF(ChosenLocation == STPFreeSlipLocation::DeviceMemory) {

						AND_WHEN("Memory was allocated on device") {

							THEN("A piece of host read-only cache can also be retrieved") {
								//host memory buffer can be retrieved as well when we are on device mode, the buffer is read only
								TestType* HostCache = (*TestBuffer)(STPFreeSlipLocation::HostMemory);
								this->compareHostDeviceBuffer(HostCache, Mode);
							}

						}
					}

				}

				AND_THEN("Manipulation on the texture data and disintegrating the data repects the memory mode") {
					const auto index = GENERATE(take(2, random(0u, CurrentTester::SmallSize - 1u)));
					const TestType RandomData = CurrentTester::getRandomData();
					TestType* const OperatedMergedBuffer = this->getBuffer();

					CHECKED_IF(Mode == TestMemoryMode::ReadOnly) {
						//test if data is available
						MERGED_AVAILABLE();

						//write something to the texture and clean up manually
						WRITE_MERGED();

						//make sure the original data is intact under read only mode
						REQUIRE(this->Reference[index] == this->Texture[index]);
					}
					CHECKED_IF(Mode == TestMemoryMode::WriteOnly) {
						//checking if the merged texture contains garbage data
						//reading garbage data is a undefined behaviour, so good luck
						const bool result = OperatedMergedBuffer[index] == this->Texture[index];
						CHECK_FALSE(result);
						CHECKED_IF(result) {
							WARN("This assertion involves reading un-initialised memory, re-run this test case a few more times to confirm");
						}

						WRITE_MERGED();

						//the new data should be written back
						TEXTURE_WRITTEN();
					}
					CHECKED_IF(Mode == TestMemoryMode::ReadWrite) {
						MERGED_AVAILABLE();

						WRITE_MERGED();

						TEXTURE_WRITTEN();
					}
				}
			}

		}

	}

}

template<typename T>
class LocalIndexRef : protected STPFreeSlipGenerator {
private:

	unique_ptr<STPFreeSlipData> HostDataCache;

protected:

	constexpr static uvec2 Dimension = uvec2(8u, 4u);
	constexpr static uvec2 ChunkUnit = uvec2(4u, 8u);
	constexpr static uvec2 ChunkRange = Dimension * ChunkUnit;

	inline static STPFreeSlipTextureAttribute IndexAttribute{ Dimension.x * Dimension.y, STPPinnedMemoryPool(), 0 };
	constexpr static typename STPFreeSlipTextureBuffer<T>::STPFreeSlipTextureData IndexData
		{ 1u, STPFreeSlipTextureBuffer<T>::STPFreeSlipTextureData::STPMemoryMode::ReadOnly, 0 };

	const STPFreeSlipData* prepareData(const STPFreeSlipManager<T>& manager, STPFreeSlipLocation location) {
		const auto* rawdata = manager.Data;

		if (location == STPFreeSlipLocation::DeviceMemory) {
			//if the manager data is on device we need to copy it back to host before we can use it
			this->HostDataCache = make_unique<STPFreeSlipData>();
			STPcudaCheckErr(cudaMemcpy(this->HostDataCache.get(), rawdata, sizeof(STPFreeSlipData), cudaMemcpyDeviceToHost));

			return this->HostDataCache.get();
		}
		return rawdata;
	}

public:

	T Local[ChunkRange.y][ChunkRange.x];
	vector<T*> LocalBuffer;

	LocalIndexRef() : Local(), STPFreeSlipGenerator(LocalIndexRef::ChunkUnit, LocalIndexRef::Dimension) {

		for (unsigned int y = 0u; y < ChunkRange.y; y++) {
			for (unsigned int x = 0u; x < ChunkRange.x; x++) {
				//generate a simple texture with local indices
				this->Local[y][x] = static_cast<T>(x + y * ChunkRange.x);
			}
			//push the texture of this chunk
			this->LocalBuffer.emplace_back(Local[y]);
		}

		static bool AttrInit = false;
		if (!AttrInit) {
			LocalIndexRef::IndexAttribute.DeviceMemPool = STPTestInformation::TestDeviceMemoryPool;
			AttrInit = true;
		}

	}

	T locate(const uvec2& coordinate) const {
		return this->Local[coordinate.y][coordinate.x];
	}

};

TEMPLATE_TEST_CASE_METHOD(LocalIndexRef, "STPFreeSlipGenerator generates global-local index table and exports data to STPFreeSlipManagerAdaptor",
	"[GPGPU][FreeSlip][STPFreeSlipGenerator]", float, Sample) {

	WHEN("Invalid data is given to the generator") {
		constexpr uvec2 BadChunkUnit = uvec2(56u, 0u);
		constexpr uvec2 BadDimension = uvec2(8u);

		THEN("Generator should not be created with error thrown") {
			REQUIRE_THROWS_AS(STPFreeSlipGenerator(BadChunkUnit, BadDimension), STPException::STPBadNumericRange);
		}

	}

	GIVEN("A freshly created, valid free-slip generator") {

		THEN("Data stored in the generator should be available upon constrction") {
			REQUIRE(this->getDimension() == LocalIndexRef::Dimension);
			REQUIRE(this->getFreeSlipChunk() == LocalIndexRef::ChunkUnit);
			REQUIRE(this->getFreeSlipRange() == LocalIndexRef::ChunkRange);
		}

		AND_GIVEN("A loaded free-slip texture buffer") {
			STPFreeSlipTextureBuffer<TestType> TextureBuffer(this->LocalBuffer, LocalIndexRef::IndexData, LocalIndexRef::IndexAttribute);

			WHEN("A free-slip adaptor is requested") {
				auto Adaptor = (*this)(TextureBuffer);

				AND_WHEN("A free-slip manager is generated from the adaptor") {
					const STPFreeSlipLocation ChosenLocation = GENERATE(values({
						STPFreeSlipLocation::HostMemory,
						STPFreeSlipLocation::DeviceMemory
					}));
					auto Manager = Adaptor(ChosenLocation);
					
					THEN("Data included in the manager should be consistent") {
						const auto* PreparedData = this->prepareData(Manager, ChosenLocation);

						REQUIRE(PreparedData->Dimension == LocalIndexRef::Dimension);
						REQUIRE(PreparedData->FreeSlipChunk == LocalIndexRef::ChunkUnit);
						REQUIRE(PreparedData->FreeSlipRange == LocalIndexRef::ChunkRange);

						CHECKED_IF(ChosenLocation == STPFreeSlipLocation::HostMemory) {
							//we don't need to check the device texture since we have tested free-slip texture buffer
							//which guarantees the texture on device is the same as host

							AND_THEN("Texture can be indexed correctly using global-local index table, and the correctness of index table is verified") {
								const auto IndexXY = GENERATE(take(5, chunk(2, random(0u, 31u))));
								const uvec2 Coordinate = glm::make_vec2(IndexXY.data());
								const unsigned int CoordinateLocal = static_cast<unsigned int>(this->locate(Coordinate));

								//index table correctness, our texture simply converts 2D coordinate to 1D index
								//when texture is flatten in the manager, the relationship is simply:
								//Texture[Index] = Index ==Implies==> Manager[Local] == Texture[Manager(Local)]
								REQUIRE(Manager[CoordinateLocal] == static_cast<TestType>(Manager(CoordinateLocal)));
								//symmetric, convert to global index and then back to local index
								REQUIRE(Manager[static_cast<unsigned int>(Manager[CoordinateLocal])] == static_cast<TestType>(CoordinateLocal));
							}

						}

					}

				}

			}

		}

	}

}