#include <SuperTerrain+/World/Diversity/STPLayer.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <algorithm>

#include <cstring>

using std::make_unique;

using namespace SuperTerrainPlus::STPDiversity;

/**
 * @brief Check if the number is power of 2.
 * @param val The value to check.
 * @return True if the number is power of 2.
*/
inline static bool isPow2(const unsigned long long val) noexcept {
	return val && !(val & (val - 1ull));
}

//TODO: can turn it into consteval in C++ 20.
/**
 * @brief Get the mask value by evaluating power of 2 and minus one.
 * @param Bit The power to raise.
 * @return The mask value.
*/
template<unsigned long long Bit>
inline constexpr static unsigned long long getMask() noexcept {
	//1 << bits is equivalent to pow(2, bits)
	if constexpr (Bit >= 64ull) {
		return ~0;
	} else {
		return (1ull << Bit) - 1ull;
	}
}

/**
 * @brief Hash the coordinate and generate a unique hash value.
 * @param x The x world coordinate.
 * @param y The y world coordinate.
 * @param z The z world coordinate.
 * @return The unique hash value for this coordinate.
*/
inline static unsigned long long uniqueHash(const int x, const int y, const int z) noexcept {
	unsigned long long hash = static_cast<unsigned long long>(x) & getMask<28ull>();
	hash |= (static_cast<unsigned long long>(z) & getMask<28ull>()) << 28ull;
	hash |= (static_cast<unsigned long long>(y) & getMask<8ull>()) << 56ull;
	return hash;
}

/**
 * @brief An algorithm to convert key value to a raw index value in order to locate the sample value in the hash table.
 * This algorithm is based on *MurmurHash3 64-bit*.
 * @param key The value of the key.
 * @return The raw index value, it's not the same as index, do remember to limit the range of the index using mask.
*/
inline static unsigned long long mixKey(unsigned long long key) noexcept {
	key ^= key >> 33ull;
	key *= 0xFF51AFD7ED558CCDull;
	key ^= key >> 33ull;
	key *= 0xC4CEB9FE1A85EC53ull;
	key ^= key >> 33ull;
	return key;
}

STPLayer::STPLayerCache::STPLayerCache(STPLayer& layer, const size_t capacity) :
	Layer(layer), Key(make_unique<unsigned long long[]>(capacity)), Value(make_unique<Sample[]>(capacity)),
	Mask(capacity - 1ull) {
	STP_ASSERTION_NUMERIC_DOMAIN(isPow2(capacity), "The capacity of the layer cache must be a power of 2");
	//make sure the hash table starts at 0
	this->clear();
}

Sample STPLayer::STPLayerCache::get(const int x, const int y, const int z) {
	//calculate the key
	const unsigned long long key = uniqueHash(x, y, z);
	//locate the index in our hash table
	const unsigned long long index = mixKey(key) & this->Mask;

	if (this->Key[index] == key) {
		//cache hit, return the value directly
		return this->Value[index];
	}
	//cache miss, compute
	const Sample sample = this->Layer.sample(x, y, z);
	//store cache entry
	this->Key[index] = key;
	this->Value[index] = sample;
	return sample;
}

void STPLayer::STPLayerCache::clear() noexcept {
	const unsigned long long capacity = this->capacity();
	//to avoid hash collision for key (it's equivalent to -1 with signed integer)
	std::memset(this->Key.get(), 0xFFu, sizeof(unsigned long long) * capacity);
	std::memset(this->Value.get(), 0x00u, sizeof(Sample) * capacity);
}

unsigned long long STPLayer::STPLayerCache::capacity() const noexcept {
	return this->Mask + 1ull;
}

STPLayer::STPLocalSampler::STPLocalSampler(const Seed layer_seed, const Seed local_seed) noexcept :
	LayerSeed(layer_seed), LocalSeed(local_seed) {

}

Sample STPLayer::STPLocalSampler::nextValue(const Sample range) const noexcept {
	//Please do not use standard library rng, it will trash the performance
	
	//the original Minecraft implementation uses floorMod, which produces the same result as "%" when both inputs have the same sign
	const Sample val = static_cast<Sample>((this->LocalSeed >> 24ull) % static_cast<Seed>(range));

	//advance to the next sequence
	this->LocalSeed = STPLayer::mixSeed(this->LocalSeed, this->LayerSeed);
	return val;
}

Sample STPLayer::STPLocalSampler::choose(const Sample var1, const Sample var2) const noexcept {
	return this->nextValue(2u) == 0u ? var1 : var2;
}

Sample STPLayer::STPLocalSampler::choose(const Sample var1, const Sample var2, const Sample var3, const Sample var4) const noexcept {
	const Sample i = this->nextValue(4u);
	return i == 0u ? var1 : i == 1u ? var2 : i == 2u ? var3 : var4;
}

STPLayer::STPLayer(const size_t cache_size, const Seed global_seed, const Seed salt, const STPAscendantInitialiser ascendant) :
	AscendantCount(ascendant.size()), Ascendant(this->AscendantCount == 0u ? nullptr : make_unique<STPLayer*[]>(this->AscendantCount)),
	Salt(salt), LayerSeed(STPLayer::seedLayer(global_seed, this->Salt)) {
	std::transform(ascendant.begin(), ascendant.end(), this->Ascendant.get(), [](auto asc) { return &asc.get(); });

	//create a cache
	if (cache_size > 0u) {
		this->Cache.emplace(*this, cache_size);
	}
}

Seed STPLayer::seedLayer(const Seed global_seed, const Seed salt) noexcept {
	Seed midSalt = STPLayer::mixSeed(salt, salt);
	midSalt = STPLayer::mixSeed(midSalt, midSalt);
	midSalt = STPLayer::mixSeed(midSalt, midSalt);
	Seed layer_seed = STPLayer::mixSeed(global_seed, midSalt);
	layer_seed = STPLayer::mixSeed(layer_seed, midSalt);
	layer_seed = STPLayer::mixSeed(layer_seed, midSalt);
	return layer_seed;
}

Seed STPLayer::seedLocal(const int x, const int z) const noexcept {
	Seed local_seed = STPLayer::mixSeed(this->LayerSeed, x);
	local_seed = STPLayer::mixSeed(local_seed, z);
	local_seed = STPLayer::mixSeed(local_seed, x);
	local_seed = STPLayer::mixSeed(local_seed, z);
	return local_seed;
}

size_t STPLayer::cacheSize() const noexcept {
	//check for nullptr
	return this->Cache ? this->Cache->capacity() : 0u ;
}

STPLayer::STPLocalSampler STPLayer::createLocalSampler(const Seed local_seed) const noexcept {
	return STPLayer::STPLocalSampler(this->LayerSeed, local_seed);
}

STPLayer::STPLocalSampler STPLayer::createLocalSampler(const int x, const int z) const noexcept {
	return this->createLocalSampler(this->seedLocal(x, z));
}

Seed STPLayer::mixSeed(Seed s, const long long fac) noexcept {
	s *= s * 6364136223846793005ull + 1442695040888963407ull;
	s += fac;
	return s;
}

Sample STPLayer::retrieve(const int x, const int y, const int z) {
	if (!this->Cache) {
		//this layer has no cache
		return this->sample(x, y, z);
	}
	//this layer has cache assigned
	//try to read the data from the cache
	return this->Cache->get(x, y, z);
}

STPLayer& STPLayer::getAscendant(const size_t index) noexcept {
	return *this->Ascendant[index];
}

bool STPLayer::isMerging() const noexcept {
	return this->AscendantCount > 1u;
}