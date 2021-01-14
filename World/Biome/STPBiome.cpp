#include "STPBiome.h"

using namespace SuperTerrainPlus::STPBiome;

STPBiome::STPBiome() {
	//this->BiomeSettings.reset(nullptr);
}

STPBiome::STPBiome(const STPSettings::STPBiomeSettings& props) {
	this->updateProperties(props);
}

STPBiome::~STPBiome() {

}

void STPBiome::updateProperties(const STPSettings::STPBiomeSettings& props) {
	//copy the settings
	this->BiomeSettings.reset(new STPSettings::STPBiomeSettings(props));
}

Sample STPBiome::getID() const {
	return this->BiomeSettings->ID;
}

string STPBiome::getName() const {
	return this->BiomeSettings->Name;
}

float STPBiome::getTemperature() const {
	return this->BiomeSettings->Temperature;
}

float STPBiome::getPrecipitation() const {
	return this->BiomeSettings->Precipitation;
}