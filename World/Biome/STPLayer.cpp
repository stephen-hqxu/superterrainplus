#include "STPLayer.h"

using namespace SuperTerrainPlus::STPBiome;

size_t STPLayer::GLOBAL_CACHESIZE = 0ull;

STPLayer::STPLocalRNG::STPLocalRNG(Seed local_seed) : LocalSeed(local_seed) {

}

STPLayer::STPLocalRNG::~STPLocalRNG() {
	
}

Sample STPLayer::STPLocalRNG::nextVal(Sample range) const {
	//TODO: feel free to use your own algorithm to generate a random number
	//Please do not use standard library rng, it will trash the performance
	Sample val = static_cast<Sample>((this->LocalSeed >> 24ull) & static_cast<unsigned long long>(range - 1u));
	//since our local seed is a constant
	static Seed modified_local_seed = this->LocalSeed;

	if (val < 0u) {
		val += range;
	}
	//advance to the next sequence
	modified_local_seed = STPSeedMixer::mixSeed(modified_local_seed, modified_local_seed);

	return val;
}

Sample STPLayer::STPLocalRNG::choose(Sample var1, Sample var2) const {
	return this->nextVal(2) == 0 ? var1 : var2;
}

Sample STPLayer::STPLocalRNG::choose(Sample var1, Sample var2, Sample var3, Sample var4) const {
	const Sample i = this->nextVal(4);
	return i == 0 ? var1 : i == 1 ? var2 : i == 2 ? var3 : var4;
}

STPLayer::~STPLayer() {
	for (unsigned int i = 0; i < this->Ascendant.size(); i++) {
		STPLayer* parent = this->Ascendant[i];

		//tell the parent that the child is going to be deleted
		parent->ReferenceCount--;
		if (parent->ReferenceCount == 0u) {
			//delete its ascendants recursively (if any)
			//the parent has no more reference, delete
			delete parent;
		}
	}
	this->Ascendant.clear();

	//delete the cache
	delete this->Cache;
}

Seed STPLayer::genLayerSeed(Seed global_seed, Seed salt) {
	Seed midSalt = STPSeedMixer::mixSeed(salt, salt);
	midSalt = STPSeedMixer::mixSeed(midSalt, midSalt);
	midSalt = STPSeedMixer::mixSeed(midSalt, midSalt);
	Seed layer_seed = STPSeedMixer::mixSeed(global_seed, midSalt);
	layer_seed = STPSeedMixer::mixSeed(layer_seed, midSalt);
	layer_seed = STPSeedMixer::mixSeed(layer_seed, midSalt);
	return layer_seed;
}

Seed STPLayer::genLocalSeed(int x, int z) {
	Seed local_seed = STPSeedMixer::mixSeed(this->LayerSeed, x);
	local_seed = STPSeedMixer::mixSeed(local_seed, z);
	local_seed = STPSeedMixer::mixSeed(local_seed, x);
	local_seed = STPSeedMixer::mixSeed(local_seed, z);
	return local_seed;
}

void STPLayer::destroy(STPLayer* layer) {
	//we explicitly define a function to delete the pointer
	//since the layer is created usign create() but not new, it may cause confusion to other programmers if we force them to use delete() function.
	delete layer;
}

size_t STPLayer::cacheSize() {
	return this->Cache->getCapacity();
}

void STPLayer::setCache(size_t capacity) {
	STPLayer::GLOBAL_CACHESIZE = capacity;
}

size_t STPLayer::getCache() {
	return STPLayer::GLOBAL_CACHESIZE;
}

STPLayer::STPLocalRNG STPLayer::getRNG(Seed local_seed) {
	return STPLayer::STPLocalRNG(local_seed);
}

Sample STPLayer::sample_cached(int x, int y, int z) {
	//pass the layer sampling function to cache, so it will compute it when necessary
	return this->Cache->cache(x, y, z, [this](int x, int y, int z) -> Sample {
		return this->sample(x, y, z);
	});
}

STPLayer* const STPLayer::getAscendant(unsigned int index) {
	return this->getAscendantCount() != 0 && index < this->getAscendantCount() ? this->Ascendant[index] : nullptr;
}

STPLayer* const STPLayer::getAscendant() {
	return this->getAscendant(0);
}

size_t STPLayer::getAscendantCount() {
	return this->Ascendant.size();
}

bool STPLayer::isMerging() {
	return this->getAscendantCount() > 1;
}

bool STPLayer::hasAscendant() {
	return this->getAscendantCount() == 0;
}