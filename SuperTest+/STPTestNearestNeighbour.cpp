//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Generators
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/World/Chunk
#include <SuperTerrain+/World/Chunk/STPNearestNeighbourTextureBuffer.h>

#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
//Error
#include <cuda_runtime.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//Shared Test Data
#include "STPTestInformation.h"

//GLM
#include <glm/vec2.hpp>
#include <glm/gtc/type_ptr.hpp>

//Container
#include <array>
#include <algorithm>
#include <optional>

using namespace SuperTerrainPlus;
using STPDiversity::Sample;

using glm::uvec2;

using std::array;
using std::make_optional;
using std::all_of;

typedef STPNearestNeighbourTextureBufferMemoryMode NNTBMM;

template<typename T, NNTBMM MM>
class NNBufferTester {
protected:

	constexpr static STPNearestNeighbourInformation NNInfo = {
		uvec2(2u),
		uvec2(3u),
		uvec2(6u)
	};
	constexpr static unsigned int PixelCount = NNInfo.MapSize.x * NNInfo.MapSize.y,
		NNCount = NNInfo.ChunkNearestNeighbour.x * NNInfo.ChunkNearestNeighbour.y,
		MergedSize = NNInfo.TotalMapSize.x * NNInfo.TotalMapSize.y;

	typedef array<array<T, PixelCount>, NNCount> ChunkTexture_t;
	typedef array<T, MergedSize> MergedTexture_t;

	//reference data as ground truth
	constexpr static ChunkTexture_t ZeroChunkData = { };
	const ChunkTexture_t ChunkTextureRef;
	const MergedTexture_t MergedTextureRef;

private:

	static ChunkTexture_t generateChunkTextureReference() {
		ChunkTexture_t ChunkTex = { };
		//pixel value is the chunk local index
		for (unsigned int chk = 0u; chk < ChunkTex.size(); chk++) {
			auto& CurrentChunk = ChunkTex[chk];
			std::fill(CurrentChunk.begin(), CurrentChunk.end(), static_cast<T>(chk));
		}
		return ChunkTex;
	}

	static MergedTexture_t generateMergedTextureReference() {
		MergedTexture_t MergedTex = { };
		for (unsigned int y = 0u; y < NNBufferTester::NNInfo.TotalMapSize.y; y++) {
			for (unsigned int x = 0u; x < NNBufferTester::NNInfo.TotalMapSize.x; x++) {
				const uvec2 PixelCoord = uvec2(x, y),
					ChunkCoord = PixelCoord / NNBufferTester::NNInfo.MapSize;

				MergedTex[x + y * NNBufferTester::NNInfo.TotalMapSize.x] = 
					static_cast<T>(ChunkCoord.x + ChunkCoord.y * NNBufferTester::NNInfo.ChunkNearestNeighbour.x);
			}
		}
		return MergedTex;
	}

protected:

	using NNBuffer = STPNearestNeighbourTextureBuffer<T, MM>;
	using NNMemoryLocation = typename NNBuffer::STPMemoryLocation;
	using NNMergedBuffer = typename NNBuffer::STPMergedBuffer;

	static MergedTexture_t backupDeviceMergedBuffer(const NNMergedBuffer& buffer) {
		MergedTexture_t Merged;

		//if it's a device memory, we need to copy it to host before we can use it
		STP_CHECK_CUDA(cudaMemcpyAsync(Merged.data(), buffer.getDevice(), NNBufferTester::MergedSize * sizeof(T), cudaMemcpyDeviceToHost, 0));
		STP_CHECK_CUDA(cudaStreamSynchronize(0));

		return Merged;
	}

	inline void copyToDeviceMergedBuffer(const NNMergedBuffer& output) const {
		//copy the destination device buffer with reference data
		STP_CHECK_CUDA(cudaMemcpyAsync(output.getDevice(), this->MergedTextureRef.data(), NNBufferTester::MergedSize * sizeof(T), cudaMemcpyHostToDevice, 0));
		STP_CHECK_CUDA(cudaStreamSynchronize(0));
	}

	inline void copyToHostMergedBuffer(const NNMergedBuffer& output) const {
		std::copy(this->MergedTextureRef.cbegin(), this->MergedTextureRef.cend(), output.getHost());
	}

	inline bool compareChunkData(const ChunkTexture_t& chunk_data) const {
		return std::equal(chunk_data.cbegin(), chunk_data.cend(), this->ChunkTextureRef.cbegin());
	}

	inline static bool isChunkDataAllZero(const ChunkTexture_t& chunk_data) {
		return all_of(chunk_data.cbegin(), chunk_data.cend(), [](const auto& chk) {
			return all_of(chk.cbegin(), chk.cend(), [](const auto pix) { return pix == static_cast<T>(0); });
		});
	}

	inline static bool isMergedDataAllZero(const T* const merged_data) {
		return all_of(merged_data, merged_data + NNBufferTester::MergedSize, [](const auto val) { return val == static_cast<T>(0); });
	}

	inline static bool isMergedDataAllZero(const MergedTexture_t& merged_data) {
		return NNBufferTester::isMergedDataAllZero(merged_data.data());
	}

public:

	NNBufferTester() : ChunkTextureRef(NNBufferTester::generateChunkTextureReference()),
		MergedTextureRef(NNBufferTester::generateMergedTextureReference()) {

	}

};

//we could not test arbitrary combination of type and memory mode because the template is explicitly instantiated

TEMPLATE_TEST_CASE_METHOD_SIG(NNBufferTester, "STPNearestNeighbourTextureBuffer can handle merged texture request for neighbour chunk "
	"logic using different memory mode on arbitrary memory location", "[Chunk][STPNearestNeighbourTextureBuffer]",
	((typename T, NNTBMM MM), T, MM), (Sample, NNTBMM::ReadOnly), (float, NNTBMM::WriteOnly), (float, NNTBMM::ReadWrite)) {
	using CurrentTester = NNBufferTester<T, MM>;

	GIVEN("A range of nearest neighbour chunks") {
		//initialise the nearest-neighbour texture buffer and fill in with our initial zero data
		//this initial data might be overwritten depends on the memory mode
		CurrentTester::ChunkTexture_t TestData = CurrentTester::ZeroChunkData;
		T* TestDataRaw[CurrentTester::NNCount];
		//extract it into raw pointer array
		std::transform(TestData.begin(), TestData.end(), TestDataRaw, [](auto& chk) { return chk.data(); });
		//create a texture buffer master
		const CurrentTester::NNBuffer TestTextureBuffer(TestDataRaw, CurrentTester::NNInfo, { STPTestInformation::TestDeviceMemoryPool, 0 });

		//test for different memory location
		const auto ChosenLocation = GENERATE(values({
			CurrentTester::NNMemoryLocation::HostMemory,
			CurrentTester::NNMemoryLocation::DeviceMemory
		}));
		WHEN("Texture from all nearest neighbour chunks are merged into a large buffer") {
			//we use a optional to control the lifetime of the merged buffer
			auto TestDataMergedMaster = make_optional<const CurrentTester::NNBuffer::STPMergedBuffer>(TestTextureBuffer, ChosenLocation);
			const auto& TestDataMerged = *TestDataMergedMaster;

			THEN("The memory of merged buffer should be consistent with the original content and memory mode specification") {
				const bool isDevice = ChosenLocation == CurrentTester::NNMemoryLocation::DeviceMemory;
				//the comparison logic is the following
				//read only: merged buffer data is original data, preserve original data afterwards.
				//write only: merged buffer data is undefined, discard original data afterwards.
				//read write: merged buffer is original data, discard original data afterwards.

				CHECKED_IF(isDevice) {
					//copy device to host
					const auto CopiedMergedDevice = this->backupDeviceMergedBuffer(TestDataMerged);
					//we can examine the data if it is write only, because data are undefined
					if constexpr (MM != NNTBMM::WriteOnly) {
						//examine the content of merged buffer, we filled in all zeros initially
						REQUIRE(CurrentTester::isMergedDataAllZero(CopiedMergedDevice));
					}
				}
				CHECKED_ELSE(isDevice) {
					//host memory can be read directly
					if constexpr (MM != NNTBMM::WriteOnly) {
						REQUIRE(CurrentTester::isMergedDataAllZero(TestDataMerged.getHost()));
					}
				}

				AND_THEN("Manipulation on the merged data and the original chunk data respects the memory mode after un-merging") {
					//change the merged data to our reference texture
					CHECKED_IF(isDevice) {
						this->copyToDeviceMergedBuffer(TestDataMerged);
					}
					CHECKED_ELSE(isDevice) {
						this->copyToHostMergedBuffer(TestDataMerged);
					}

					//kill the merged buffer, this should trigger buffer un-merging
					TestDataMergedMaster.reset();

					//check the validity of original data
					if constexpr (MM == NNTBMM::ReadOnly) {
						//should be the original data
						this->isChunkDataAllZero(TestData);
					} else {
						//should be the newly written data
						this->compareChunkData(TestData);
					}
				}

			}

		}

	}
}