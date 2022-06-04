#include <SuperTerrain+/World/Diversity/STPBiomeFactory.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

using glm::uvec2;
using glm::uvec3;
using glm::ivec3;

using std::make_unique;

using namespace SuperTerrainPlus::STPDiversity;

STPBiomeFactory::STPProductionLineCreator::STPProductionLineCreator(const STPBiomeFactory& factory) : Factory(factory) {

}

STPBiomeFactory::STPLayerManager_t STPBiomeFactory::STPProductionLineCreator::operator()() const {
	return make_unique<STPLayerManager>(this->Factory.supply());
}

STPBiomeFactory::STPBiomeFactory(uvec3 dimension) : LayerProductionLine(*this), BiomeDimension(dimension) {
	if (dimension.x == 0u || dimension.y == 0u || dimension.z == 0u) {
		throw STPException::STPBadNumericRange("No component in a dimension vector should be zero");
	}
}

STPBiomeFactory::STPBiomeFactory(uvec2 dimension) : STPBiomeFactory(uvec3(dimension.x, 1u, dimension.y)) {

}

void STPBiomeFactory::operator()(Sample* biomemap, ivec3 offset) {
	if (this->BiomeDimension.y != 1u) {
		//it's a 3D biome
		throw STPException::STPUnsupportedFunctionality("3-dimension biomemap generation is not supported");
	}

	//it's a 2D biome
	//request a production line
	STPLayerManager_t producer = this->LayerProductionLine.requestObject();

	//loop through and generate the biome map
	//why not using CUDA and do it in parallel? Because the biome layers are cached, tested and parallel performance is a piece of shit
	for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
		for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
			//calculate the map index
			const unsigned int index = x + z * this->BiomeDimension.x;
			//get the biome at given coordinate
			biomemap[index] = producer->start()->retrieve(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.z);
		}
	}

	//free the producer
	this->LayerProductionLine.returnObject(std::move(producer));
}