//DO NOT INCLUDE THIS HEADER SEPARATELY
#ifdef _STP_LAYER_H_

#include <array>
#include <algorithm>

template <class... Asc>
inline SuperTerrainPlus::STPDiversity::STPLayer::STPLayer(const size_t cache_size, const Seed global_seed, const Seed salt, Asc* ... ascendant) :
	STPLayer(sizeof...(Asc), cache_size, global_seed, salt) {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::conjunction_v<std::is_base_of<STPLayer, Asc>...>, "Only STPLayer and its children are allowed as ascendant");
	
	//copy all ascendant pointers to the internal memory
	const std::array<STPLayer*, sizeof...(Asc)> ascendantRecord = { ascendant... };
	std::copy_n(ascendantRecord.data(), this->AscendantCount, this->Ascendant.get());
}

#endif//_STP_LAYER_H_