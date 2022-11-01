#include <SuperTerrain+/World/Diversity/STPLayerCache.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

#include <cstring>

using namespace SuperTerrainPlus::STPDiversity;

using std::make_unique;
using std::make_tuple;
using std::make_optional;

/**
 * @brief Check if the number is power of 2.
 * @param val The value to check.
 * @return True if the number is power of 2.
*/
inline constexpr static bool isPow2(unsigned long long val) noexcept {
	return val && !(val & (val - 1ull));
}

/**
 * @brief Get the mask value by evaluating power of 2 and minus one.
 * @param bits The power to raise.
 * @return The mask value.
*/
inline constexpr static unsigned long long getMask(unsigned long long bits) noexcept {
	//1 << bits is equivalent to pow(2, bits)
	return bits >= 64ull ? ~0 : (1ull << bits) - 1ull;
}

/**
 * @brief Hash the coordinate and generate a unique hash value.
 * @param x The x world coordinate.
 * @param y The y world coordinate.
 * @param z The z world coordinate.
 * @return The unique hash value for this coordinate.
*/
inline constexpr static unsigned long long uniqueHash(int x, int y, int z) noexcept {
	//TODO: Feel free to implement a more elegant hash algorithm if you would like to
	unsigned long long hash = static_cast<unsigned long long>(x) & getMask(28ull);
	hash |= (static_cast<unsigned long long>(z) & getMask(28ull)) << 28ull;
	hash |= (static_cast<unsigned long long>(y) & getMask(8ull)) << 56ull;
	return hash;
}

/**
 * @brief An algorithm to convert key value to a raw index value in order to locate the sample value in the hash table.
 * @param key The value of the key.
 * @return The raw index value, it's not the same as index, do remember to limit the range of the index using mask.
*/
inline constexpr static unsigned long long mixKey(unsigned long long key) noexcept {
	//TODO: yeah feel free to change the implementation as well if you have a more elegant idea
	key ^= key >> 33ull;
	key *= 0xFF51AFD7ED558CCDull;
	key ^= key >> 33ull;
	key *= 0xC4CEB9FE1A85EC53ull;
	key ^= key >> 33ull;
	return key;
}

STPLayerCache::STPLayerCache(size_t capacity) :
	Key(make_unique<unsigned long long[]>(capacity)), Value(make_unique<Sample[]>(capacity)), Mask(capacity - 1ull) {
	if (!isPow2(capacity)) {
		throw STPException::STPBadNumericRange("The capacity must be a power of 2");
	}
	//make sure the hash table starts at 0
	this->clearCache();
}

STPLayerCache::STPCacheData STPLayerCache::read(STPCacheEntry entry) {
	//decompose the entry
	const auto [found, key, index] = entry;

	if (found) {
		//cache found, read directly
		return make_optional(this->Value[index]);
	}
	//cache not found
	return std::nullopt;
}

void STPLayerCache::write(STPCacheEntry entry, Sample sample) {
	const auto [found, key, index] = entry;

	if (!found) {
		//write data only if entry is empty
		this->Key[index] = key;
		this->Value[index] = sample;
	}
}

STPLayerCache::STPCacheEntry STPLayerCache::locate(int x, int y, int z) const {
	//calculate the key
	const unsigned long long key = uniqueHash(x, y, z);
	//locate the index in our hash table
	const unsigned long long index = mixKey(key) & this->Mask;
	//Please be aware the cache is not thread safe!
	//That's because multithread performance is 100x worse than single thread so I remove it
	const bool found = this->Key[index] == key;

	return make_tuple(found, key, index);
}

void STPLayerCache::clearCache() {
	const unsigned long long capacity = this->capacity();
	//to avoid hash collision for key (it's equivalent to -1 with signed integer)
	memset(this->Key.get(), 0xFFu, sizeof(unsigned long long) * capacity);
	memset(this->Value.get(), 0x00u, sizeof(Sample) * capacity);
}

unsigned long long STPLayerCache::capacity() const noexcept {
	return this->Mask + 1ull;
}