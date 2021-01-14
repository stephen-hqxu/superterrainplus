#pragma once

//DO NOT INCLUDE THIS HEADER SEPARATELY
#ifdef _STP_LAYER_H_
template <class... Asc>
SuperTerrainPlus::STPBiome::STPLayer::STPLayer(Seed global_seed, Seed salt, Asc*... ascendant)
	: Salt(salt), LayerSeed(STPLayer::genLayerSeed(global_seed, salt)), Ascendant{ ascendant... } {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::conjunction<std::is_base_of<STPLayer, Asc>...>::value, "Only STPLayer and its children are allowed as ascendants");

	this->ReferenceCount = 0u;
	//create cache
	this->Cache = new STPLayerCache(STPLayer::GLOBAL_CACHESIZE);
	//increment reference count for parents
	for (auto it = this->Ascendant.begin(); it != this->Ascendant.end(); it++) {
		(*it)->ReferenceCount++;
	}
}

template <class L, class... Arg>
SuperTerrainPlus::STPBiome::STPLayer* SuperTerrainPlus::STPBiome::STPLayer::create(Seed global_seed, Seed salt, Arg&&... args) {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::is_base_of<STPLayer, L>::value, "Only STPLayer and its children are allowed to instantiate as a new layer");
	
	//instantiate
	return dynamic_cast<STPLayer*>(new L(global_seed, salt, std::forward<Arg>(args)...));
}
#endif//_STP_LAYER_H_