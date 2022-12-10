#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

#include <glm/ext/scalar_constants.hpp>

using namespace SuperTerrainPlus::STPEnvironment;

//STPSimplexNoiseSetting.h

#define ASSERT_SIMPLEX_NOISE(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPSimplexNoiseSetting)

void STPSimplexNoiseSetting::validate() const {
	ASSERT_SIMPLEX_NOISE(this->Distribution != 0);
	ASSERT_SIMPLEX_NOISE(this->Offset >= 0.0);
	ASSERT_SIMPLEX_NOISE(this->Offset < 2.0 * glm::pi<double>());
}