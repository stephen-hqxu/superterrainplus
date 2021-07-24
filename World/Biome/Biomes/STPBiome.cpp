#include "STPBiome.h"

using namespace STPDemo;

using std::string;

STPBiome::STPBiome() {

}

STPBiome::STPBiome(const STPBiomeSettings& props) {
	this->updateProperties(props);
}

STPBiome::~STPBiome() {

}

void STPBiome::updateProperties(const STPBiomeSettings& props) {
	//copy the settings
	this->BiomeSettings = props;
}

const STPBiomeSettings& STPBiome::getProperties() const {
	return this->BiomeSettings;
}

Sample STPBiome::getID() const {
	return this->BiomeSettings.ID;
}

string STPBiome::getName() const {
	return this->BiomeSettings.Name;
}

float STPBiome::getTemperature() const {
	return this->BiomeSettings.Temperature;
}

float STPBiome::getPrecipitation() const {
	return this->BiomeSettings.Precipitation;
}