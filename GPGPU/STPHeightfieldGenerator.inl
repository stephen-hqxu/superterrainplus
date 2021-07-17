#pragma once

//DO NOT INCLUDE THIS HEADER MANUALLY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_HEIGHTFIELD_GENERATOR_CUH_

#include "STPDeviceErrorHandler.cuh"
//TODO: redo this. What if biome ids are interleaved?
template<typename Ite>
__host__ void SuperTerrainPlus::STPCompute::STPHeightfieldGenerator::linkDictionary(Ite begin, Ite end) {
	//Type check
	static_assert(typeid(std::iterator_traits<Ite>::value_type) == typeid(SuperTerrainPlus::STPDiversity::STPBiome*), "Iterator must go through STPBiome");
	const unsigned int count = std::distance(begin, end);

	//make sure no kernel is active right now
	STPcudaCheckErr(cudaDeviceSynchronize());
	STPDiversity::STPBiome* bd_cache;
	STPcudaCheckErr(cudaMalloc(&bd_cache, sizeof(SuperTerrainPlus::STPDiversity::STPBiome) * count));

	//copy all biomes to device in order
	unsigned int loc = 0u;
	for (const auto& it = begin; it != end; it++) {
		STPcudaCheckErr(cudaMemcpy(bd_cache + loc, *it, sizeof(SuperTerrainPlus::STPDiversity::STPBiome), cudaMemcpyHostToDevice));
		loc++;
	}

	//finish up
	STPcudaCheckErr(cudaDeviceSynchronize());
	this->BiomeDictionary = unique_ptr_d<STPDiversity::STPBiome>(bd_cache);
}

template<class Fac, typename... Arg>
__host__ void SuperTerrainPlus::STPCompute::STPHeightfieldGenerator::linkBiomeFactory(Arg&&... arg) {
	using namespace STPDiversity;
	//init a new biome factory
	STPBiomeFactory* factory = dynamic_cast<STPBiomeFactory*>(new Fac(std::forward<Arg>(arg)...));
	//assign to our heightfield generator
	this->biome = std::unique_ptr<STPBiomeFactory>(factory);
}

#endif//_STP_HEIGHTFIELD_GENERATOR_CUH_