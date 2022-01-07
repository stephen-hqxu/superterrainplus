#include <SuperTerrain+/World/Diversity/STPLayer.h>

using namespace SuperTerrainPlus::STPDiversity;

STPLayer::STPLocalRNG::STPLocalRNG(Seed layer_seed, Seed local_seed) : LayerSeed(layer_seed), LocalSeed(local_seed) {

}

STPLayer::STPLocalRNG::~STPLocalRNG() {
	
}

Sample STPLayer::STPLocalRNG::nextVal(Sample range) const {
	//TODO: feel free to use your own algorithm to generate a random number
	//Please do not use standard library rng, it will trash the performance
	
	//the original Minecraft implementation uses floorMod, which produces the same result as "%" when both inputs have the same sign
	const Sample val = static_cast<Sample>((this->LocalSeed >> 24ull) % static_cast<Seed>(range));

	//advance to the next sequence
	this->LocalSeed = STPSeedMixer::mixSeed(this->LocalSeed, this->LayerSeed);
	return val;
}

STPLayer::STPLayer(Seed global_seed, Seed salt) : Salt(salt), LayerSeed(STPLayer::genLayerSeed(global_seed, salt)) {

}

Sample STPLayer::STPLocalRNG::choose(Sample var1, Sample var2) const {
	return this->nextVal(2) == 0 ? var1 : var2;
}

Sample STPLayer::STPLocalRNG::choose(Sample var1, Sample var2, Sample var3, Sample var4) const {
	const Sample i = this->nextVal(4);
	return i == 0 ? var1 : i == 1 ? var2 : i == 2 ? var3 : var4;
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

Seed STPLayer::genLocalSeed(int x, int z) const {
	Seed local_seed = STPSeedMixer::mixSeed(this->LayerSeed, x);
	local_seed = STPSeedMixer::mixSeed(local_seed, z);
	local_seed = STPSeedMixer::mixSeed(local_seed, x);
	local_seed = STPSeedMixer::mixSeed(local_seed, z);
	return local_seed;
}

size_t STPLayer::cacheSize() const {
	//check for nullptr
	return this->Cache ? this->Cache->getCapacity() : 0ull ;
}

STPLayer::STPLocalRNG STPLayer::getRNG(Seed local_seed) const {
	return STPLayer::STPLocalRNG(this->LayerSeed, local_seed);
}

Sample STPLayer::retrieve(int x, int y, int z) {
	if (this->Cache) {
		//this layer has cache assigned
		//try to read the data from the cache
		const STPLayerCache::STPCacheEntry entry = this->Cache->locate(x, y, z);
		const STPLayerCache::STPCacheData data = this->Cache->read(entry);

		if (!data.has_value()) {
			//data is not cached, need to sample
			const Sample sample = this->sample(x, y, z);
			//and cache the new data
			this->Cache->write(entry, sample);

			//return the computed data
			return sample;
		}
		//data is cached, read it
		return *data;
	}

	//this layer has no cache
	return this->sample(x, y, z);
}

STPLayer* STPLayer::getAscendant(unsigned int index) const {
	return this->getAscendantCount() != 0 && index < this->getAscendantCount() ? this->Ascendant[index] : nullptr;
}

STPLayer* STPLayer::getAscendant() const {
	return this->getAscendant(0);
}

size_t STPLayer::getAscendantCount() const {
	return this->Ascendant.size();
}

bool STPLayer::isMerging() const {
	return this->getAscendantCount() > 1;
}

bool STPLayer::hasAscendant() const {
	return this->getAscendantCount() != 0;
}