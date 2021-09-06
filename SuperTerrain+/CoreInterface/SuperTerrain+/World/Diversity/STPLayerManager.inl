#pragma once

//DO NOT INCLUDE THIS HEADER SEPARATELY
#ifdef _STP_LAYER_MANAGER_H_
template <class L, size_t C, class... Arg>
inline SuperTerrainPlus::STPDiversity::STPLayer* SuperTerrainPlus::STPDiversity::STPLayerManager::insert(Arg&&... args) {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::is_base_of<STPLayer, L>::value, "Only STPLayer and its children are allowed to instantiate as a new layer");

	try {
		//instantiate a new layer
		//I hate using `new`, but there's no dynamic_cast for unique_ptr so I have to...
		STPLayer* newLayer = dynamic_cast<STPLayer*>(new L(std::forward<Arg>(args)...));
		//let this pointer managed by the current layer manager
		this->Vertex.emplace_back(newLayer, &STPLayerManager::recycleLayer);

		//create cache
		if (C != 0ull) {
			//user is allowed to use no cache, but that's totally at their own risk
			//assign the new cache, it might be nullptr if user didn't ask to create a cache
			newLayer->Cache = std::make_unique<STPLayerCache>(C);
		}
		return newLayer;
	}
	catch (...) {
		std::rethrow_exception(std::current_exception());
	}

}
#endif//_STP_LAYER_MANAGER_H_