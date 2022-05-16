#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

using namespace SuperTerrainPlus::STPEnvironment;

//STPSimplexNoiseSetting.h

STPSimplexNoiseSetting::STPSimplexNoiseSetting() : STPSetting(), 
	Seed(0ull), 
	Distribution(8u), 
	Offset(45.0) {

}

bool STPSimplexNoiseSetting::validate() const {
	return this->Distribution != 0
		&& this->Offset >= 0.0
		&& this->Offset < 360.0;
}