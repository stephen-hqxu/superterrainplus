//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_get_random_seed.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>

//SuperAlgorithm+Host
#include <SuperAlgorithm+Host/STPSingleHistogramFilter.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//System
#include <utility>
#include <limits>

#include <glm/vec2.hpp>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPAlgorithm;

using SuperTerrainPlus::STPDiversity::Sample;
using FiltBuf = STPSingleHistogramFilter::STPFilterBuffer;
using Exec = FiltBuf::STPExecutionType;

using glm::uvec2;

using std::pair;
using std::make_pair;

#define EXPECT_PAIR(X, Y) make_pair<Sample>(X, Y)

class HistogramTester : protected STPSingleHistogramFilter {
private:

	constexpr static uvec2 Dimension = uvec2(4u);
	constexpr static uvec2 Neighbour = uvec2(3u);
	constexpr static uvec2 TotalDimension = Dimension * Neighbour;

	//reference texture
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
		Neighbour,
		TotalDimension
	};

protected:

	constexpr static unsigned int BinOffsetCount = Dimension.x * Dimension.y + 1u;

	FiltBuf SerialBuffer, ParallelBuffer;

	inline STPSingleHistogram execute(FiltBuf& filter_buffer, const unsigned int radius = 2u) {
		return (*this)(HistogramTester::Texture, HistogramTester::Data, filter_buffer, radius);
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

	HistogramTester() : SerialBuffer(Exec::Serial), ParallelBuffer(Exec::Parallel) {

	}

};

SCENARIO_METHOD(HistogramTester, "STPSingleHistogramFilter analyses a sample texture and output histograms for every pixel within a given radius", 
	"[AlgorithmHost][STPSingleHistogramFilter]") {

	GIVEN("A single histogram filter with a reference sample-map texture") {

		WHEN("Launch a single histogram filter with wrong arguments") {

			THEN("Error should be thrown to prevent bad things to happen") {
				//radius is zero
				REQUIRE_THROWS_AS(this->execute(this->SerialBuffer, 0u), STPException::STPNumericDomainError);
				//radius is bigger than the total texture size
				REQUIRE_THROWS_AS(this->execute(this->ParallelBuffer, 128u), STPException::STPNumericDomainError);
				//radius is not an even number
				REQUIRE_THROWS_AS(this->execute(this->SerialBuffer, 3u), STPException::STPNumericDomainError);
			}

		}

		WHEN("Launch it again with the correct argument") {
			const auto ExecutionType = GENERATE(values({ Exec::Serial, Exec::Parallel }));
			const bool UseParallel = ExecutionType == Exec::Parallel;
			const char* const FilterCaseName = UseParallel ? "parallel" : "serial";
			auto& WorkingBuffer = UseParallel ? this->ParallelBuffer : this->SerialBuffer;

			AND_WHEN("The filter is launched with " << FilterCaseName << " memory buffer, implies " << FilterCaseName << " execution") {

				THEN("Histogram should be computed without errors") {
					STPSingleHistogram Result = { };
					REQUIRE_NOTHROW([this, &WorkingBuffer, &Result]() { Result = this->execute(WorkingBuffer); }());

					//all results should be the same
					AND_THEN("The output histogram should be verified in terms of correctness") {
						this->verifyHistogram(Result);

						AND_THEN("The output should be the same when the texture input is the same") {
							STPSingleHistogram anotherResult = this->execute(WorkingBuffer);
							this->verifyHistogram(anotherResult);
						}
					}

					AND_THEN("Various auxiliary query functions from the filter buffer can be called with correct output") {
						CHECK(WorkingBuffer.type() == ExecutionType);

						const auto [bin_count, offset_count] = WorkingBuffer.size();
						CHECK(offset_count == HistogramTester::BinOffsetCount);
						CHECK(bin_count == Result.HistogramStartOffset[offset_count - 1u]);
					}

				}

			}

		}

	}

}

//Nanobench
#include "../STPTestInformation.h"

#include <memory>
#include <algorithm>
#include <sstream>

//Container
#include <array>
#include <string>

#include <glm/exponential.hpp>

using std::array;
using std::generate_n;
using std::string, std::to_string, std::ostringstream;
using std::unique_ptr, std::make_unique;

using ankerl::nanobench::Bench;
using ankerl::nanobench::Rng;
using ankerl::nanobench::doNotOptimizeAway;

class HistogramBenchmarker {
private:

	constexpr static char SHFResultFilename[] = "SingleHistogramFilter";

	struct BenchmarkRunData {
	public:

		Bench& Profiler;

		STPSingleHistogramFilter& Filter;
		const Sample* const Image;
		FiltBuf& Buffer;

	};

#define EXPAND_RUN_DATA const auto& [profiler, filter, image, buffer] = run_data

	//The current problem size is determined by: initial * growth^{i} \forall i \in [0, iteration]
	constexpr static unsigned int DimensionGrowth = 2u, DimensionIteration = 6u,
		RadiusGrowth = 3u, RadiusIteration = 4u;
	constexpr static array<Sample, 4u> SampleRangeTrial = { 2u, 8u, 15u, 30u };

	constexpr static uvec2 SampleDimensionInitial = uvec2(16u),
		SampleNeighbourCount = uvec2(3u);
	constexpr static unsigned int FilterRadiusInitial = 2u;

	//The default sizes are when we are varying one parameter while fixing others to this value
	constexpr static uvec2 DefaultSampleDimension = uvec2(192u);
	constexpr static unsigned int DefaultFilterRadius = 16u;
	constexpr static Sample DefaultSampleRangeMax = 5u;

	//calculated values from the settings above
	constexpr static uvec2 DefaultSampleTotalDimension = DefaultSampleDimension * SampleNeighbourCount;
	constexpr static size_t DefaultSampleTextureSize = DefaultSampleTotalDimension.x * DefaultSampleTotalDimension.y;
	const uvec2 MaxSampleTotalDimension;
	const size_t MaxSampleTextureSize;

	static STPNearestNeighbourInformation getNeighbourInfo(const uvec2& mapSize) noexcept {
		const uvec2 total_size = mapSize * HistogramBenchmarker::SampleNeighbourCount;
		return STPNearestNeighbourInformation { mapSize, HistogramBenchmarker::SampleNeighbourCount, total_size };
	}

	static string formatDimension(const uvec2& dim) {
		ostringstream dim_str;
		dim_str << dim.x << 'x' << dim.y;
		return dim_str.str();
	}

	static string formatSampleRange(const Sample range_max) {
		ostringstream range_str;
		range_str << "[0, " << range_max << ']';
		return range_str.str();
	}

	static void varyDimension(const BenchmarkRunData& run_data) {
		EXPAND_RUN_DATA;

		uvec2 currentDim = SampleDimensionInitial;
		for (unsigned int i = 0u; i <= DimensionIteration; i++) {
			profiler.complexityN(currentDim.x * currentDim.y)
				.run(HistogramBenchmarker::formatDimension(currentDim),
					[nn = HistogramBenchmarker::getNeighbourInfo(currentDim), &filter, image, &buffer]() {
						doNotOptimizeAway(filter(image, nn, buffer, DefaultFilterRadius));
					});

			currentDim *= DimensionGrowth;
		}
	}

	static void varyRadius(const BenchmarkRunData& run_data) {
		EXPAND_RUN_DATA;
		const auto Neighbour = HistogramBenchmarker::getNeighbourInfo(DefaultSampleDimension);

		unsigned int currentRadius = FilterRadiusInitial;
		for (unsigned int i = 0u; i <= RadiusIteration; i++) {
			profiler.complexityN(currentRadius)
				.run(to_string(currentRadius), [&Neighbour, &filter, image, &buffer, currentRadius]() {
					doNotOptimizeAway(filter(image, Neighbour, buffer, currentRadius));
				});

			currentRadius *= RadiusGrowth;
		}
	}

	static void varyRange(const BenchmarkRunData& run_data, Rng& generator) {
		EXPAND_RUN_DATA;
		const auto Neighbour = HistogramBenchmarker::getNeighbourInfo(DefaultSampleDimension);

		for (const auto rangeMax : SampleRangeTrial) {
			//regenerate the test image with a different range
			generate_n(const_cast<Sample*>(image), DefaultSampleTextureSize,
				[&generator, rangeMax]() noexcept { return generator.bounded(rangeMax); });

			profiler.complexityN(rangeMax)
				.run(HistogramBenchmarker::formatSampleRange(rangeMax), [&Neighbour, &filter, image, &buffer]() {
					doNotOptimizeAway(filter(image, Neighbour, buffer, DefaultFilterRadius));
				});
		}
	}

public:

	HistogramBenchmarker() noexcept :
		MaxSampleTotalDimension(SampleNeighbourCount * SampleDimensionInitial
			* static_cast<unsigned int>(glm::pow(DimensionGrowth, DimensionIteration))),
		MaxSampleTextureSize(this->MaxSampleTotalDimension.x * this->MaxSampleTotalDimension.y) {
	
	}

	~HistogramBenchmarker() = default;

	void operator()() {
		//benchmark configuration
		auto ResultFile = STPTestInformation::createBenchmarkResultFile(SHFResultFilename);
		Rng SampleGenerator(Catch::getSeed());
		Bench SHFProfiler;
		//warm up to get the adaptive memory pool going
		{
			using namespace std::chrono_literals;
			SHFProfiler.output(&ResultFile).unit("filtering").timeUnit(1ms, "ms").maxEpochTime(500ms).warmup(1ull);
		}

		const auto exportResult = [&SHFProfiler, &ResultFile](
				const string& execution_type, const char* const variable) -> void {
			ostringstream shf_result_name;
			shf_result_name << SHFResultFilename << '-' << execution_type << '-' << variable;

			STPTestInformation::renderBenchmarkResult(shf_result_name.str().c_str(), SHFProfiler);

			//print complexity
			ResultFile << SHFProfiler.complexityBigO() << '\n' << std::endl;
		};

		/* ---------------------------------- allocate random sample texture -------------------------- */
		const unique_ptr<Sample[]> TestSampleMemory = make_unique<Sample[]>(MaxSampleTextureSize);
		Sample* const TestSample = TestSampleMemory.get();

		using namespace std::string_literals;
		//create filter buffer
		STPSingleHistogramFilter SHFFilter;
		auto ExecutionBuffer =
			array { make_pair(FiltBuf(Exec::Serial), "Serial"s), make_pair(FiltBuf(Exec::Parallel), "Parallel"s) };

		/* --------------------------- investigate impact of dimension and radius ---------------------- */
		//to save time, we generate the maximum size texture, and use sub-texture in each run
		generate_n(TestSample, MaxSampleTextureSize,
			[&SampleGenerator]() noexcept { return SampleGenerator.bounded(DefaultSampleRangeMax); });
		SHFProfiler.context("Sample Range", HistogramBenchmarker::formatSampleRange(DefaultSampleRangeMax));

#define LOOP_EXECUTION for (auto& [buffer, executionType] : ExecutionBuffer)
#define PREPARE_RUN_DATA \
const HistogramBenchmarker::BenchmarkRunData run_data { SHFProfiler, SHFFilter, TestSample, buffer }; \
SHFProfiler.context("Execution", executionType)

		//we run different problem sizes, and use 2 execution strategy: serial and parallel
		LOOP_EXECUTION {
			//prepare benchmark data
			PREPARE_RUN_DATA;

			//dimension
			SHFProfiler.title("Dimension::" + executionType)
				.context("Dimension", "Vary")
				.context("Radius", to_string(DefaultFilterRadius));
			REQUIRE_NOTHROW(HistogramBenchmarker::varyDimension(run_data));
			exportResult(executionType, "Dimension");

			//radius
			SHFProfiler.title("Radius::" + executionType)
				.context("Radius", "Vary")
				.context("Dimension", HistogramBenchmarker::formatDimension(DefaultSampleDimension));
			REQUIRE_NOTHROW(HistogramBenchmarker::varyRadius(run_data));
			exportResult(executionType, "Radius");
		}

		/* -------------------------- investigate impact of sample range on the image -------------------- */
		SHFProfiler.context("Sample Range", "Vary")
			.context("Dimension", HistogramBenchmarker::formatDimension(DefaultSampleDimension))
			.context("Radius", to_string(DefaultFilterRadius));
		LOOP_EXECUTION {
			PREPARE_RUN_DATA;
			
			SHFProfiler.title("Sample Range::" + executionType);
			REQUIRE_NOTHROW(HistogramBenchmarker::varyRange(run_data, SampleGenerator));
			exportResult(executionType, "SampleRange");
		}
	}

};
METHOD_AS_TEST_CASE(HistogramBenchmarker::operator(), "STPSingleHistogramFilter Benchmark", "[AlgorithmHost][STPSingleHistogramFilter][!benchmark]");