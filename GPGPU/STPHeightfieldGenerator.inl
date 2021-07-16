#pragma once

//DO NOT INCLUDE THIS HEADER MANUALLY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_HEIGHTFIELD_GENERATOR_CUH_
#include "STPDeviceErrorHandler.cuh"
//TODO: redo this. What if biome ids are interleaved?
template<typename Ite>
__host__ void SuperTerrainPlus::STPCompute::STPHeightfieldGenerator::defineDictionary(Ite begin, Ite end) {
	//Type check
	static_assert(typeid(std::iterator_traits<Ite>::value_type) == typeid(SuperTerrainPlus::STPDiversity::STPBiome*), "Iterator must go through STPBiome");
	const unsigned int count = std::distance(begin, end);

	//make sure no kernel is active right now
	STPcudaCheckErr(cudaDeviceSynchronize());
	if (this->BiomeDictionary != nullptr) {
		//free the original storage and resize
		STPcudaCheckErr(cudaFree(this->BiomeDictionary));
	}
	STPcudaCheckErr(cudaMalloc(&this->BiomeDictionary, sizeof(SuperTerrainPlus::STPDiversity::STPBiome) * count));

	//copy all biomes to device in order
	unsigned int loc = 0u;
	for (const auto& it = begin; it != end; it++) {
		STPcudaCheckErr(cudaMemcpy(this->BiomeDictionary + loc, *it, sizeof(SuperTerrainPlus::STPDiversity::STPBiome), cudaMemcpyHostToDevice));
		loc++;
	}

	//finish up
	STPcudaCheckErr(cudaDeviceSynchronize());
}

#endif//_STP_HEIGHTFIELD_GENERATOR_CUH_