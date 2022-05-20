#pragma once

//DO NOT INCLUDE THIS HEADER SEPARATELY
#ifdef _STP_LAYER_H_
template <class... Asc>
inline SuperTerrainPlus::STPDiversity::STPLayer::STPLayer(Seed global_seed, Seed salt, Asc*... ascendant)
	: Ascendant{ ascendant... }, Salt(salt), LayerSeed(STPLayer::genLayerSeed(global_seed, salt)) {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::conjunction<std::is_base_of<STPLayer, Asc>...>::value, "Only STPLayer and its children are allowed as ascendant");
}
#endif//_STP_LAYER_H_