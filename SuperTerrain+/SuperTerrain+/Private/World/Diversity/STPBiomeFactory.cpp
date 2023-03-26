#include <SuperTerrain+/World/Diversity/STPBiomeFactory.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

using glm::uvec2;
using glm::ivec2;

using std::make_unique;

using namespace SuperTerrainPlus::STPDiversity;

STPBiomeFactory::STPProductionLineCreator::STPProductionLineCreator(STPBiomeFactory& factory) noexcept : Factory(factory) {

}

STPLayer* STPBiomeFactory::STPProductionLineCreator::operator()() {
	return &this->Factory.supply();
}

STPBiomeFactory::STPBiomeFactory(const uvec2 dimension) : LayerProductionLine(*this), BiomeDimension(dimension) {
	STP_ASSERTION_NUMERIC_DOMAIN(dimension.x > 0u && dimension.y > 0u, "Biomemap should have strictly positive dimension in both vector components");
}

void STPBiomeFactory::operator()(Sample* const biomemap, const ivec2 offset) {
	//request a production line
	STPLayer& tree = *this->LayerProductionLine.request();

	//y-component is interpreted as z coordinate in world space

	//loop through and generate the biome map
	for (unsigned int z = 0u; z < this->BiomeDimension.y; z++) {
		for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
			//calculate the map index
			const unsigned int index = x + z * this->BiomeDimension.x;
			//get the biome at given coordinate
			biomemap[index] = tree.retrieve(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.y);
		}
	}

	//free the producer
	this->LayerProductionLine.release(&tree);
}