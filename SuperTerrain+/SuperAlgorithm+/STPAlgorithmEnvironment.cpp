#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

#include <glm/ext/scalar_constants.hpp>

using namespace SuperTerrainPlus::STPEnvironment;

//STPSimplexNoiseSetting.h

STPSimplexNoiseSetting::STPSimplexNoiseSetting() : STPSetting(), 
	Seed(0ull), 
	Distribution(8u), 
	Offset(glm::pi<double>() * 0.25) {

}

bool STPSimplexNoiseSetting::validate() const {
	return this->Distribution != 0
		&& this->Offset >= 0.0
		&& this->Offset < 2.0 * glm::pi<double>();
}