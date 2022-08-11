//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/GPGPU/FreeSlip
#include <SuperTerrain+/World/Chunk/STPFreeSlipTextureBuffer.h>

#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//Shared Test Data
#include "STPTestInformation.h"

#include <glm/gtc/type_ptr.hpp>

#include <type_traits>
#include <algorithm>
#include <optional>
#include <random>

using namespace SuperTerrainPlus;

using STPDiversity::Sample;

using glm::uvec2;

using std::vector;
using std::optional;
using std::unique_ptr;
using std::make_unique;
using std::make_pair;

template<typename T>
class FreeSlipBufferTester {
protected:

	using MemoryLocation = typename STPFreeSlipTextureBuffer<T>::STPFreeSlipLocation;

private:

	T* CurrentMergedTexture;
	//only for device memory location
	T* MergedDevice;
	unique_ptr<T[]> MergedHost;

	MemoryLocation CurrentLocation;

protected:

	constexpr static STPFreeSlipInformation SmallInfo = {
		uvec2(8u),
		uvec2(1u),
		uvec2(8u)
	};
	constexpr static unsigned int SmallSize = SmallInfo.FreeSlipRange.x * SmallInfo.FreeSlipRange.y;

	using CurrentFreeSlipBuffer = STPFreeSlipTextureBuffer<T>;
	using TestData = typename CurrentFreeSlipBuffer::STPFreeSlipTextureData;
	using TestMemoryMode = typename TestData::STPMemoryMode;

	T Reference[SmallSize];
	T Texture[SmallSize];
	vector<T*> TextureBuffer;

	inline static STPFreeSlipTextureAttribute SmallAttribute{ SmallInfo, STPPinnedMemoryPool(), 0 };

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

	void registerMergedBuffer(MemoryLocation location, T* buffer) {
		this->CurrentMergedTexture = buffer;
		this->CurrentLocation = location;

		if (this->CurrentLocation == MemoryLocation::DeviceMemory) {
			typedef FreeSlipBufferTester<T> FBT;
			//if it's a device memory, we need to copy it to host before we can use it
			this->MergedDevice = this->CurrentMergedTexture;
			this->MergedHost = make_unique<T[]>(FBT::SmallSize);
			STP_CHECK_CUDA(cudaMemcpyAsync(this->MergedHost.get(), this->MergedDevice, FBT::SmallSize * sizeof(T), cudaMemcpyDeviceToHost, 0));
			STP_CHECK_CUDA(cudaStreamSynchronize(0));

			//reassign pointer so later assertions can use it on host side
			this->CurrentMergedTexture = MergedHost.get();
		}
	}

	inline void compareHostDeviceBuffer(T* host_cache, TestMemoryMode mode) {
		REQUIRE_FALSE(host_cache == this->MergedDevice);
		CHECKED_ELSE(mode == TestMemoryMode::WriteOnly) {
			//we can't guarantee the data on both ends is the same on write only mode
			REQUIRE(std::equal(host_cache, host_cache + FreeSlipBufferTester<T>::SmallSize, this->MergedHost.get()));
		}
	}

	inline T* getBuffer() {
		return this->CurrentMergedTexture;
	}

	inline void updateDeviceBuffer() {
		if (this->CurrentLocation == MemoryLocation::DeviceMemory) {
			STP_CHECK_CUDA(cudaMemcpyAsync(this->MergedDevice, this->CurrentMergedTexture, FreeSlipBufferTester<T>::SmallSize * sizeof(T), cudaMemcpyHostToDevice, 0));
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
	using CurrentTester = ::FreeSlipBufferTester<TestType>;

	WHEN("Some wrong numbers are provided to the texture buffer") {
		STPFreeSlipTextureAttribute ZeroPixel = { CurrentTester::SmallInfo, STPPinnedMemoryPool(), 0 };
		typename CurrentTester::TestData DefaultData = { CurrentTester::TestMemoryMode::ReadOnly, 0 };

		AND_WHEN("The number of free-slip texture does not logically match the free-slip setting") {
			vector<TestType*> EmptyBuffer;
			vector<TestType*> ALotBuffer(8u, this->Texture);

			THEN("Construction of free-slip texture buffer should be also prevented") {
				//no texture buffer
				REQUIRE_THROWS_AS(typename CurrentTester::CurrentFreeSlipBuffer(EmptyBuffer, DefaultData, CurrentTester::SmallAttribute), STPException::STPInvalidArgument);
				//more texture buffer are provided for free-slip
				REQUIRE_THROWS_AS(typename CurrentTester::CurrentFreeSlipBuffer(ALotBuffer, DefaultData, CurrentTester::SmallAttribute), STPException::STPInvalidArgument);
			}

		}

	}

	GIVEN("A valid array of texture buffer and appropriate memory operation mode") {
		const typename CurrentTester::TestMemoryMode Mode = GENERATE(values({
			CurrentTester::TestMemoryMode::ReadOnly,
			CurrentTester::TestMemoryMode::WriteOnly,
			CurrentTester::TestMemoryMode::ReadWrite
		}));
		optional<typename CurrentTester::CurrentFreeSlipBuffer> TestBuffer;

		REQUIRE_NOTHROW([&TestBuffer, Mode, this]() {
			typename CurrentTester::TestData Data = { Mode, 0 };
			TestBuffer.emplace(this->TextureBuffer, Data, CurrentTester::SmallAttribute);
		}());

		WHEN("Texture buffer is un-merged") {

			THEN("Merge location is not available") {
				REQUIRE_THROWS_AS(TestBuffer->where() == CurrentTester::MemoryLocation::HostMemory, STPException::STPMemoryError);
			}

		}

		WHEN("Merge the texture with said memory mode") {
			const typename CurrentTester::MemoryLocation ChosenLocation = GENERATE(values({
				CurrentTester::MemoryLocation::HostMemory,
				CurrentTester::MemoryLocation::DeviceMemory
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

					CHECKED_IF(ChosenLocation == CurrentTester::MemoryLocation::DeviceMemory) {

						AND_WHEN("Memory was allocated on device") {

							THEN("A piece of host read-only cache can also be retrieved") {
								//host memory buffer can be retrieved as well when we are on device mode, the buffer is read only
								TestType* HostCache = (*TestBuffer)(CurrentTester::MemoryLocation::HostMemory);
								this->compareHostDeviceBuffer(HostCache, Mode);
							}

						}
					}

				}

				AND_THEN("Manipulation on the texture data and disintegrating the data respects the memory mode") {
					const auto index = GENERATE(take(2, random(0u, CurrentTester::SmallSize - 1u)));
					const TestType RandomData = CurrentTester::getRandomData();
					TestType* const OperatedMergedBuffer = this->getBuffer();

					CHECKED_IF(Mode == CurrentTester::TestMemoryMode::ReadOnly) {
						//test if data is available
						MERGED_AVAILABLE();

						//write something to the texture and clean up manually
						WRITE_MERGED();

						//make sure the original data is intact under read only mode
						REQUIRE(this->Reference[index] == this->Texture[index]);
					}
					CHECKED_IF(Mode == CurrentTester::TestMemoryMode::WriteOnly) {
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
					CHECKED_IF(Mode == CurrentTester::TestMemoryMode::ReadWrite) {
						MERGED_AVAILABLE();

						WRITE_MERGED();

						TEXTURE_WRITTEN();
					}
				}
			}

		}

	}

}