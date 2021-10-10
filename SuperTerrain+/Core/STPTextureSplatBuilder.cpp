#include <SuperTerrain+/World/Diversity/Texture/STPTextureSplatBuilder.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::make_tuple;
using std::make_pair;

template<class S, class M>
STPTextureSplatBuilder::STPStructureView<S> STPTextureSplatBuilder::visitMapping(const M& mapping) {
	STPStructureView<S> table(mapping.size());
	//copy the mapping to array
	std::transform(mapping.cbegin(), mapping.cend(), table.begin(), [](const auto& mapping) {
		return make_pair(mapping.first, const_cast<const S*>(&(mapping.second)));
	});

	return table;
}

const STPTextureSplatBuilder::STPAltitudeStructure& STPTextureSplatBuilder::getAltitude(Sample sample) const {
	auto it = this->BiomeAltitudeMapping.find(sample);
	if (it == this->BiomeAltitudeMapping.cend()) {
		throw STPException::STPMemoryError("No altitude structure is associated with said sample");
	}
	return it->second;
}

STPTextureSplatBuilder::STPAltitudeView STPTextureSplatBuilder::visitAltitude() const {
	return STPTextureSplatBuilder::visitMapping<STPAltitudeStructure>(this->BiomeAltitudeMapping);
}

const STPTextureSplatBuilder::STPGradientStructure& STPTextureSplatBuilder::getGradient(Sample sample) const {
	auto it = this->BiomeGradientMapping.find(sample);
	if (it == this->BiomeGradientMapping.cend()) {
		throw STPException::STPMemoryError("No gradient structure is associated with said sample");
	}
	return it->second;
}

STPTextureSplatBuilder::STPGradientView STPTextureSplatBuilder::visitGradient() const {
	return STPTextureSplatBuilder::visitMapping<STPGradientStructure>(this->BiomeGradientMapping);
}

void STPTextureSplatBuilder::addAltitude(Sample sample, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//get the biome altitude mapping; create one if it does not exist yet
	STPAltitudeStructure& alt = this->BiomeAltitudeMapping[sample];
	//same, replace the old texture ID if exists
	STPTextureInformation::STPAltitudeNode& altNode = alt[upperBound];
	altNode = { { }, upperBound };
	//TODO: we can use designated initialiser to init the base union in C++20
	altNode.Reference.DatabaseKey = texture_id;
}

void STPTextureSplatBuilder::addGradient
	(Sample sample, float minGradient, float maxGradient, float lowerBound, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//find gradient, or insert
	STPGradientStructure& gra = this->BiomeGradientMapping[sample];

	//create the gradient data
	STPTextureInformation::STPGradientNode newData = { { }, minGradient, maxGradient, lowerBound, upperBound };
	//TODO: the same here
	newData.Reference.DatabaseKey = texture_id;
	gra.emplace_back(newData);
}
