#include <SuperTerrain+/World/Diversity/Texture/STPTextureSplatBuilder.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::make_tuple;
using std::make_pair;

const STPTextureSplatBuilder::STPAltitudeStructure& STPTextureSplatBuilder::getAltitude(Sample sample) const {
	auto it = this->STPBiomeAltitudeMapping.find(sample);
	if (it == this->STPBiomeAltitudeMapping.cend()) {
		throw STPException::STPMemoryError("No altitude structure is associated with said sample");
	}
	return it->second;
}

const STPTextureSplatBuilder::STPGradientStructure& STPTextureSplatBuilder::getGradient(Sample sample) const {
	auto it = this->STPBiomeGradientMapping.find(sample);
	if (it == this->STPBiomeGradientMapping.cend()) {
		throw STPException::STPMemoryError("No gradient structure is associated with said sample");
	}
	return it->second;
}

void STPTextureSplatBuilder::addAltitude(Sample sample, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//get the biome altitude mapping; create one if it does not exist yet
	STPAltitudeStructure& alt = this->STPBiomeAltitudeMapping[sample];
	//same, replace the old texture ID if exists
	alt[upperBound] = texture_id;
}

void STPTextureSplatBuilder::addGradient
	(Sample sample, float minGradient, float maxGradient, float lowerBound, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//find gradient, or insert
	STPGradientStructure& gra = this->STPBiomeGradientMapping[sample];

	//create the gradient data
	STPGradientData newData = make_tuple(minGradient, maxGradient, lowerBound, upperBound);
	gra.emplace_back(make_pair(newData, texture_id));
}
