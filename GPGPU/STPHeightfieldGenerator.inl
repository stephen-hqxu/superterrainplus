#pragma once

//DO NOT INCLUDE THIS HEADER MANUALLY, IT'S AUTOMATICALLY MANAGED
#ifdef _STP_HEIGHTFIELD_GENERATOR_CUH_

template<typename Ite>
__host__ bool SuperTerrainPlus::STPCompute::STPHeightfieldGenerator::defineDictionary(Ite begin, Ite end) {
	//Type check
	static_assert(typeid(std::iterator_traits<Ite>::value_type) == typeid(SuperTerrainPlus::STPBiome::STPBiome*), "Iterator must go through STPBiome");
	const unsigned int count = std::distance(begin, end);

	bool success = true;
	cudaStream_t copy_stream;
	success &= cudaSuccess == cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
	if (this->BiomeDictionary != nullptr) {
		//free the original storage and resize
		success &= cudaSuccess == cudaFree(this->BiomeDictionary);
	}
	success &= cudaSuccess == cudaMalloc(&this->BiomeDictionary, sizeof(SuperTerrainPlus::STPBiome::STPBiome) * count);

	//copy all biomes to device in order
	unsigned int loc = 0u;
	for (const auto& it = begin; it != end; it++) {
		success &= cudaSuccess == cudaMemcpyAsync(this->BiomeDictionary[loc], *it, sizeof(SuperTerrainPlus::STPBiome::STPBiome), cudaMemcpyHostToDevice, copy_stream);
		loc++;
	}

	success &= cudaSuccess == cudaStreamSynchronize(copy_stream);
	success &= cudaSuccess == cudaStreamDestroy(copy_stream);

	return success;
}

#endif//_STP_HEIGHTFIELD_GENERATOR_CUH_