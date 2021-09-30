#include <SuperTerrain+/World/Diversity/Texture/STPTextureSplatBuilder.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>

#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::make_tuple;
using std::make_pair;

template<class S, class M>
inline STPTextureSplatBuilder::STPStructureView<S> STPTextureSplatBuilder::sortMapping(const M& mapping) const {
	STPStructureView<S> table(mapping.size());
	//copy the mapping to array
	std::transform(mapping.cbegin(), mapping.cend(), table.begin(), [](const auto& alt) {
		return make_pair(alt.first, const_cast<const S*>(&(alt.second)));
	});

	//sort the table
	std::sort(table.begin(), table.end(), [](const auto& v1, const auto& v2) {
		return v1.first < v2.first;
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

STPTextureSplatBuilder::STPAltitudeView STPTextureSplatBuilder::sortAltitude() const {
	return this->sortMapping<STPAltitudeStructure>(this->BiomeAltitudeMapping);
}

const STPTextureSplatBuilder::STPGradientStructure& STPTextureSplatBuilder::getGradient(Sample sample) const {
	auto it = this->BiomeGradientMapping.find(sample);
	if (it == this->BiomeGradientMapping.cend()) {
		throw STPException::STPMemoryError("No gradient structure is associated with said sample");
	}
	return it->second;
}

STPTextureSplatBuilder::STPGradientView STPTextureSplatBuilder::sortGradient() const {
	return this->sortMapping<STPGradientStructure>(this->BiomeGradientMapping);
}

void STPTextureSplatBuilder::addAltitude(Sample sample, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//get the biome altitude mapping; create one if it does not exist yet
	STPAltitudeStructure& alt = this->BiomeAltitudeMapping[sample];
	//same, replace the old texture ID if exists
	alt[upperBound] = texture_id;
}

void STPTextureSplatBuilder::addGradient
	(Sample sample, float minGradient, float maxGradient, float lowerBound, float upperBound, STPTextureDatabase::STPTextureID texture_id) {
	//find gradient, or insert
	STPGradientStructure& gra = this->BiomeGradientMapping[sample];

	//create the gradient data
	STPGradientData newData = make_tuple(minGradient, maxGradient, lowerBound, upperBound);
	gra.emplace_back(make_pair(newData, texture_id));
}
