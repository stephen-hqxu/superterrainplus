#include <World/Diversity/STPBiomeFactory.h>

#include <Utility/Exception/STPUnsupportedFunctionality.h>

using glm::uvec2;
using glm::uvec3;
using glm::ivec3;

using namespace SuperTerrainPlus::STPDiversity;

STPBiomeFactory::STPBiomeFactory(uvec3 dimension) : BiomeDimension(dimension) {

}

STPBiomeFactory::STPBiomeFactory(uvec2 dimension) : STPBiomeFactory(uvec3(dimension.x, 1u, dimension.y)) {

}

STPBiomeFactory::STPLayerManager_t STPBiomeFactory::requestProductionLine() {
	{
		std::unique_lock lock(this->ProductionLock);
		STPLayerManager_t line;

		if (this->LayerProductionLine.empty()) {
			//no more idling line? Create a new one
			line = STPLayerManager_t(this->supply());
			return line;
		}
		//otherwise simply pop from the idling queue
		line = move(this->LayerProductionLine.front());
		this->LayerProductionLine.pop();
		return line;
	}
}

void STPBiomeFactory::returnProductionLine(STPLayerManager_t& line) {
	{
		std::unique_lock lock(this->ProductionLock);
		//simply put it back
		this->LayerProductionLine.emplace(move(line));
	}
}

void STPBiomeFactory::operator()(Sample* biomemap, ivec3 offset) {
	//request a production line
	STPLayerManager_t producer = this->requestProductionLine();

	//loop through and generate the biome map
	//why not using CUDA and do it in parallel? Because the biome layers are cached, tested and parallel performance is a piece of shit
	if (this->BiomeDimension.y == 1u) {
		//it's a 2D biome
		//to avoid making useless computation
		for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
			for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
				//calculate the map index
				const unsigned int index = x + z * this->BiomeDimension.x;
				//get the biome at thie coordinate
				biomemap[index] = producer->start()->sample(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.z);
			}
		}
		return;
	}
	//it's a 3D biome
	throw STPException::STPUnsupportedFunctionality("3-dimension biomemap generation is not supported");

	//free the producer
	this->returnProductionLine(producer);
}