//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//SuperAlgorithm+Host
#include <SuperAlgorithm+Host/STPPermutationGenerator.h>

//Exceptions
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

#include <cuda_runtime.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//System
#include <memory>
#include <limits>

#include <glm/trigonometric.hpp>

namespace Err = SuperTerrainPlus::STPException;
namespace PermGen = SuperTerrainPlus::STPAlgorithm::STPPermutationGenerator;
using SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting;

using std::unique_ptr;
using std::make_unique;

using glm::radians;

struct SimplexArg : public STPSimplexNoiseSetting {
public:

	SimplexArg() : STPSimplexNoiseSetting() {
		this->Seed = 6666ull;
		this->Distribution = 10u;
		this->Offset = radians(6.5);
	}

};

class PermutationGenTester {
public:

	unique_ptr<unsigned char[]> HostPerm;
	unique_ptr<float[]> HostGrad2D;

protected:

	const SimplexArg Args;
	const PermGen::STPPermutationTable Result;

	void copyTable() {
		//copy device table back to host
		const auto& table_d = this->Result.Permutation;
		//allocation
		this->HostPerm = make_unique<unsigned char[]>(512);
		this->HostGrad2D = make_unique<float[]>(table_d.Gradient2DSize * 2);
		//copy
		STP_CHECK_CUDA(cudaMemcpy(this->HostPerm.get(), table_d.Permutation, sizeof(unsigned char) * 512, cudaMemcpyDeviceToHost));
		STP_CHECK_CUDA(cudaMemcpy(this->HostGrad2D.get(), table_d.Gradient2D, sizeof(float) * table_d.Gradient2DSize * 2, cudaMemcpyDeviceToHost));
	}

public:

	PermutationGenTester() : Args(), Result(PermGen::generate(this->Args)) {
		this->copyTable();
	}

	PermutationGenTester(const SimplexArg args) : Args(args), Result(PermGen::generate(this->Args)) {
		this->copyTable();
	}

};

//it depends on the precision of built-in trig function
#define FLOAT_EQUAL(X) Catch::Matchers::WithinRel(X, std::numeric_limits<float>::epsilon() * 10.0f)
#define GET_GRADX(D) glm::cos(radians(D))
#define GET_GRADY(D) glm::sin(radians(D))
#define CHECK_GRAD(I, D) CHECK_THAT(this->HostGrad2D[I], FLOAT_EQUAL(GET_GRADX(D))); \
CHECK_THAT(this->HostGrad2D[I + 1], FLOAT_EQUAL(GET_GRADY(D)))

SCENARIO_METHOD(PermutationGenTester, "STPPermutationGenerator can a generate deterministically random permutation table", 
	"[AlgorithmHost][STPPermutationGenerator]") {

	GIVEN("A simplex noise setting with invalid values") {
		STPSimplexNoiseSetting InvalidArg;
		InvalidArg.Distribution = 0u;

		THEN("Construction of permutation generator should be prevented") {
			REQUIRE_THROWS_AS(PermGen::generate(InvalidArg), Err::STPInvalidEnvironment);
		}

	}

	GIVEN("A correct simplex noise setting") {

		WHEN("Permutation is retrieved from generation") {
			const auto& permutation = this->Result.Permutation;

			THEN("Correctness of the gradient table should be verified") {
				REQUIRE(permutation.Gradient2DSize == this->Args.Distribution);

				//pick some gradients and check
				CHECK_GRAD(0, 6.5f);
				CHECK_GRAD(6, 114.5f);
				CHECK_GRAD(14, 258.5f);
			}

			THEN("Permutation table should be repeated correctly") {
				const unsigned int baseIndex = GENERATE(take(3, random(0u, 255u)));
				REQUIRE(this->HostPerm[baseIndex] == this->HostPerm[baseIndex + 256u]);
			}

			AND_GIVEN("Another permutation table") {
				SimplexArg AnotherArg = this->Args;

				WHEN("The new permutation table has the same seed value as the current one") {

					THEN("The two tables should be equal") {
						//the same seed
						PermutationGenTester AnotherGen(AnotherArg);
						const unsigned int Index = GENERATE(take(3, random(0u, 255u)));
						REQUIRE(AnotherGen.HostPerm[Index] == this->HostPerm[Index]);
					}

				}

				WHEN("The new permutation table has a different seed value than the current one") {

					THEN("The two permutation tables should be distinct") {
						//random seed
						AnotherArg.Seed = 183457092ull;
						PermutationGenTester AnotherGen(AnotherArg);
						//due to its randomness, we can't always guarantee two values are different
						auto Indices = GENERATE(take(3, chunk(3, random(0u, 255u))));
						unsigned int notEq_count = 0u;
						for (int i = 0; i < 3; i++) {
							//pick 3 values, and if any of that is different, we count as "distinct"
							const unsigned int Index = Indices[i];
							notEq_count += (AnotherGen.HostPerm[Index] == this->HostPerm[Index]) ? 0u : 1u;
						}
						CHECK_FALSE(notEq_count == 0u);
					}
				}

			}

		}

	}

}