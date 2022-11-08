#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

#include <glm/ext/scalar_constants.hpp>

using namespace SuperTerrainPlus::STPEnvironment;

//STPSimplexNoiseSetting.h

void STPSimplexNoiseSetting::validate() const {
	if (this->Distribution != 0
		&& this->Offset >= 0.0
		&& this->Offset < 2.0 * glm::pi<double>()) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPSimplexNoiseSetting validation fails");
}