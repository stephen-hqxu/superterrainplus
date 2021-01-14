#include "STPLayerCache.h"

using namespace SuperTerrainPlus::STPBiome;

bool STPLayerCache::isPow2(unsigned long long val) {
	return val && !(val & (val - 1ull));
}

unsigned long long STPLayerCache::getMask(unsigned long long bits) {
	//1 << bits is equivalent to pow(2, bits)
	return bits >= 64ull ? ~0 : (1ull << bits) - 1ull;
}

unsigned long long STPLayerCache::uniqueHash(int x, int y, int z) {
	//TODO: Feel free to implement a more elegant hash algorithm if you would like to
	unsigned long long hash = static_cast<unsigned long long>(x) & STPLayerCache::getMask(28ull);
	hash |= (static_cast<unsigned long long>(z) & STPLayerCache::getMask(28ull)) << 28ull;
	hash |= (static_cast<unsigned long long>(y) & STPLayerCache::getMask(8ull)) << 56ull;
	return hash;
}

unsigned long long STPLayerCache::mixKey(unsigned long long key) {
	//TODO: yeah feel free to change the implementation as well if you have a more elegant idea
	key ^= key >> 33ull;
	key *= 0xFF51AFD7ED558CCDull;
	key ^= key >> 33ull;
	key *= 0xC4CEB9FE1A85EC53ull;
	key ^= key >> 33ull;
	return key;
}

STPLayerCache::STPLayerCache(size_t capacity) {
	if (!STPLayerCache::isPow2(capacity)) {
		throw std::invalid_argument("The capacity must be a power of 2");
	}
	
	this->Mask = capacity - 1ull;
	//allocate space for cache and clear the storage
	this->Key = new unsigned long long[capacity];
	this->Value = new Sample[capacity];
	this->clearCache();

}

STPLayerCache::~STPLayerCache() {
	delete[] this->Key;
	delete[] this->Value;
}

Sample STPLayerCache::cache(int x, int y, int z, std::function<Sample(int, int, int)> sampler) {
	//calc the key
	const unsigned long long key = STPLayerCache::uniqueHash(x, y, z);
	//locate the index in our hash table
	const unsigned long long index = STPLayerCache::mixKey(key) & this->Mask;

	{
		std::shared_lock<std::shared_mutex> read_safe(this->lock);
		if (this->Key[index] == key) {
			//cache found, read it directly
			return this->Value[index];
		}
	}
	{	
		std::unique_lock<std::shared_mutex> write_safe(this->lock);
		//cache not found, compute then store
		Sample sample = sampler(x, y, z);
		this->Key[index] = key;
		this->Value[index] = sample;
		return sample;
	}
	
}

void STPLayerCache::clearCache() {
	const size_t capacity = this->getCapacity();
	memset(this->Key, 0x00, sizeof(unsigned long long) * capacity);
	memset(this->Value, 0x00, sizeof(Sample) * capacity);
}

size_t STPLayerCache::getCapacity() {
	return this->Mask + 1ull;
}