#pragma once
#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

using namespace SuperTerrainPlus::STPEnvironment;

STPSimplexNoiseSetting::STPSimplexNoiseSetting() : STPSetting() {
	//Loading default value
	this->Seed = 0u;
	this->Distribution = 8u;
	this->Offset = 45.0;
}

bool STPSimplexNoiseSetting::validate() const {
	return this->Distribution != 0
		&& this->Offset >= 0.0
		&& this->Offset < 360.0;
}