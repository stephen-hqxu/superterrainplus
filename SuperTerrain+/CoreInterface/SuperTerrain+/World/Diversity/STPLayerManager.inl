//DO NOT INCLUDE THIS HEADER SEPARATELY
#ifdef _STP_LAYER_MANAGER_H_

template <class L, class... Arg>
SuperTerrainPlus::STPDiversity::STPLayer* SuperTerrainPlus::STPDiversity::STPLayerManager::insert(size_t cache_size, Arg&&... args) {
	//make sure only STPLayer is supplied as template, error throws at compile time
	static_assert(std::is_base_of<STPLayer, L>::value, "Only STPLayer and its children are allowed to instantiate as a new layer");
	using std::unique_ptr;
	using std::make_unique;
	using std::move;

	//instantiate a new layer
	unique_ptr<L> targetLayer = make_unique<L>(cache_size, std::forward<Arg>(args)...);
	//cast it to pointer to the base layer
	unique_ptr<STPLayer> genericLayer(static_cast<STPLayer*>(targetLayer.release()));

	//add to the layer library
	return this->Vertex.emplace_back(move(genericLayer)).get();
}

inline SuperTerrainPlus::STPDiversity::STPLayer* SuperTerrainPlus::STPDiversity::STPLayerManager::start() const noexcept {
	return this->Vertex.back().get();
}

inline size_t SuperTerrainPlus::STPDiversity::STPLayerManager::getLayerCount() const noexcept {
	return this->Vertex.size();
}

#endif//_STP_LAYER_MANAGER_H_