//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//SuperAlgorithm+Host
#include <SuperAlgorithm+/STPSingleHistogramFilter.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//System
#include <utility>
#include <limits>

#include <glm/vec2.hpp>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPAlgorithm;

using SuperTerrainPlus::STPDiversity::Sample;

using glm::uvec2;
using std::pair;
using std::make_pair;

#define EXPECT_PAIR(X, Y) make_pair<Sample>(X, Y)

class HistogramTester : protected STPSingleHistogramFilter {
private:

	constexpr static uvec2 Dimension = uvec2(4u);
	constexpr static uvec2 Unit = uvec2(3u);
	constexpr static uvec2 Range = Dimension * Unit;

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

	constexpr static STPNearestNeighbourInformation Data = {
		Dimension,
		Unit,
		Range
	};

protected:

	STPSingleHistogramFilter::STPFilterBuffer Buffer = STPSingleHistogramFilter::STPFilterBuffer();

	inline STPSingleHistogram execute(const unsigned int radius = 2u) {
		return (*this)(HistogramTester::Texture, HistogramTester::Data, this->Buffer, radius);
	}

	void verifyHistogram(const STPSingleHistogram& result) {
		//verify it using human brain...
		constexpr static unsigned int TrialOffset[] = { 0u, 8u, 15u };
		constexpr static pair<Sample, float> Expected[] = {
			//0-1
			EXPECT_PAIR(0u, 0.24f),
			EXPECT_PAIR(3u, 0.24f),
			EXPECT_PAIR(2u, 0.28f),
			EXPECT_PAIR(1u, 0.24f),
			//8-9
			EXPECT_PAIR(0u, 0.36f),
			EXPECT_PAIR(3u, 0.24f),
			EXPECT_PAIR(1u, 0.16f),
			EXPECT_PAIR(2u, 0.24f),
			//14-15
			EXPECT_PAIR(0u, 0.32f),
			EXPECT_PAIR(3u, 0.2f),
			EXPECT_PAIR(1u, 0.12f),
			EXPECT_PAIR(2u, 0.36f)
		};
		//loop
		for (int trial = 0; trial < 3; trial++) {
			const auto& offset = TrialOffset[trial];

			for (auto [i, counter] = make_pair(result.HistogramStartOffset[offset], 0); i < result.HistogramStartOffset[offset + 1u]; i++, counter++) {
				const auto [item, weight] = Expected[counter + trial * 4];
				REQUIRE(result.Bin[i].Item == item);
				//because our numbers are weights, which are always less than one
				REQUIRE_THAT(result.Bin[i].Weight, Catch::Matchers::WithinRel(weight, std::numeric_limits<float>::epsilon() * 5.0f));
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
				REQUIRE_THROWS_AS(this->execute(0u), STPException::STPNumericDomainError);
				//radius is bigger than the free-slip texture
				REQUIRE_THROWS_AS(this->execute(128u), STPException::STPNumericDomainError);
				//radius is not an even number
				REQUIRE_THROWS_AS(this->execute(3u), STPException::STPNumericDomainError);
			}

		}

		WHEN("Launch it again with the correct argument") {

			THEN("Histogram should be computed without errors") {
				STPSingleHistogram result = { };
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

		}

	}

}