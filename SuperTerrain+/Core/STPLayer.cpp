#include <World/Diversity/STPLayer.h>

#include <functional>

using namespace SuperTerrainPlus::STPDiversity;

STPLayer::STPLocalRNG::STPLocalRNG(Seed local_seed) : LocalSeed(local_seed) {

}

STPLayer::STPLocalRNG::~STPLocalRNG() {
	
}

Sample STPLayer::STPLocalRNG::nextVal(Sample range) const {
	//TODO: feel free to use your own algorithm to generate a random number
	//Please do not use standard library rng, it will trash the performance
	static auto floorMod = [](Seed x, Seed y) -> Sample {
		return static_cast<Sample>(x - (static_cast<Seed>(__floor(1.0 * x / y * 1.0)) * y));
	};
	//since our local seed is a constant
	static Seed modified_local_seed = this->LocalSeed;
	Sample val = floorMod(this->LocalSeed >> 24ull, static_cast<unsigned long long>(range));

	if (val < 0u) {
		val += range;
	}
	//advance to the next sequence
	modified_local_seed = STPSeedMixer::mixSeed(modified_local_seed, modified_local_seed);

	return val;
}

SuperTerrainPlus::STPDiversity::STPLayer::STPLayer(Seed global_seed, Seed salt) : Salt(salt), LayerSeed(STPLayer::genLayerSeed(global_seed, salt)) {

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
	return STPLayer::STPLocalRNG(local_seed);
}

Sample STPLayer::retrieve(int x, int y, int z) {
	using namespace std::placeholders;
	auto sampler = std::bind(&STPLayer::sample, this, _1, _2, _3);
	//pass the layer sampling function to cache, so it will compute it when necessary
	if (this->Cache) {
		return this->Cache->cache(x, y, z, sampler);
	}
	//there is no cache assigned
	return sampler(x, y, z);
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