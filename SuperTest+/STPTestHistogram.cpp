//Catch2
#include <catch2/catch_test_macros.hpp>

//SuperAlgorithm+Host
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//System
#include <utility>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;

using SuperTerrainPlus::STPDiversity::Sample;

using glm::uvec2;
using std::pair;
using std::make_pair;

class SampleTexture {
private:

	constexpr static uvec2 Dimension = uvec2(4u);
	constexpr static uvec2 Unit = uvec2(3u);
	constexpr static uvec2 Range = uvec2(12u);

	constexpr static unsigned int PixelCount = Range.x * Range.y;

	constexpr static Sample Texture[] = {
		2, 0, 2, 0, 1, 1, 0, 3, 3, 2, 1, 2,
		0, 1, 2, 1, 1, 1, 3, 2, 2, 0, 0, 1,
		0, 1, 0, 2, 1, 3, 1, 1, 1, 2, 1, 0,
		3, 3, 0, 1, 1, 2, 2, 2, 2, 0, 0, 3,
		3, 2, 3, 3, 0, 3, 2, 2, 1, 0, 0, 3,
		2, 0, 0, 1, 2, 0, 2, 2, 0, 2, 0, 3,
		1, 2, 3, 0, 3, 2, 1, 2, 2, 3, 0, 2,
		1, 1, 0, 3, 0, 2, 1, 0, 3, 2, 2, 1,
		2, 2, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1,
		0, 1, 3, 3, 0, 3, 3, 3, 1, 0, 1, 1,
		3, 1, 3, 0, 2, 1, 1, 0, 2, 2, 2, 0,
		2, 0, 1, 1, 2, 3, 1, 2, 3, 2, 0, 1
	};

	unsigned int IndexTable[PixelCount];

	STPFreeSlipData Data;

public:

	//manager does no construt-time check and it only retains a pointer, we are free to init the data later
	STPFreeSlipSampleManager Manager = STPFreeSlipSampleManager(const_cast<Sample*>(Texture), &Data);

	SampleTexture() {
		//init the index table
		for (unsigned int i = 0u; i < PixelCount; i++) {
			this->IndexTable[i] = i;
		}

		//init the data
		this->Data = {
			this->IndexTable,
			SampleTexture::Dimension,
			SampleTexture::Unit,
			SampleTexture::Range
		};
	}

};

class HistogramTester : protected STPSingleHistogramFilter {
private:

	SampleTexture Texture;

protected:

	inline const static STPSingleHistogramFilter::STPHistogramBuffer_t Buffer = STPSingleHistogramFilter::createHistogramBuffer();

	inline STPSingleHistogram execute(unsigned int radius = 2u) {
		return (*this)(this->Texture.Manager, Buffer, radius);
	}

	void verifyHistogram(const STPSingleHistogram& result) {
		//verify it using human brain...
		constexpr static unsigned int TrialOffset[] = { 0u, 8u, 15u };
		constexpr static pair<Sample, float> Expected[] = {
			//0-1
			make_pair(0u, 0.24f),
			make_pair(3u, 0.24f),
			make_pair(2u, 0.28f),
			make_pair(1u, 0.24f),
			//8-9
			make_pair(0u, 0.36f),
			make_pair(3u, 0.24f),
			make_pair(1u, 0.16f),
			make_pair(2u, 0.24f),
			//14-15
			make_pair(0u, 0.32f),
			make_pair(3u, 0.2f),
			make_pair(1u, 0.12f),
			make_pair(2u, 0.36f),

		};
		//loop
		for (int trial = 0; trial < 3; trial++) {
			const auto& offset = TrialOffset[trial];

			for (auto [i, counter] = make_pair(result.HistogramStartOffset[offset], 0); i < result.HistogramStartOffset[offset + 1u]; i++, counter++) {
				const auto& currEV = Expected[counter + trial * 4];
				REQUIRE(result.Bin[i].Item == currEV.first);
				REQUIRE(result.Bin[i].Data.Weight == currEV.second);
			}
		}
	}

public:

	HistogramTester() : STPSingleHistogramFilter() {

	}

};

SCENARIO_METHOD(HistogramTester, "STPSingleHistogramFilter analyses a sample texture and output histograms for every pixel within a given radius", 
	"[AlgorithmHost][STPSingleHistogramFilter]") {

	GIVEN("An initialised single histogram filter with a fixed histogram buffer") {

		WHEN("Launch a single histogram filter with wrong arguments") {

			THEN("Error should be thrown to prevent bad things to happen") {
				//radius is zero
				REQUIRE_THROWS_AS(this->execute(0u), STPException::STPBadNumericRange);
				//radius is bigger than the free-slip texture
				REQUIRE_THROWS_AS(this->execute(128u), STPException::STPBadNumericRange);
				//radius is not an even number
				REQUIRE_THROWS_AS(this->execute(3u), STPException::STPBadNumericRange);
			}

		}

		WHEN("Launch it again with the correct argument") {

			THEN("Histogram should be computed without errors") {
				STPSingleHistogram result;
				REQUIRE_NOTHROW([this, &result]() { result = this->execute(); }());

				//all results should be the same
				AND_THEN("The output histogram should be verified in terms of correctness") {
					this->verifyHistogram(result);

					AND_THEN("The output should be the same when the texture input is the same") {
						STPSingleHistogram anotherResult = this->execute();
						this->verifyHistogram(anotherResult);
					}
				}

			}

			THEN("The histogram result can be retrieved even though the generator is dead") {
				//retrieve the histogram directly from the buffer
				//the buffer is static in the test class so it won't be reset
				auto readResult = this->readHistogramBuffer(HistogramTester::Buffer);
				this->verifyHistogram(readResult);
			}

		}

	}

}