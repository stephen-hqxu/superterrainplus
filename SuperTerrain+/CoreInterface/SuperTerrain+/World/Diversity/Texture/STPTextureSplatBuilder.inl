//PLEASE DO NOT INCLUDE THIE INLINE DEFINITION AS IT'S MANAGED AUTOMATICALLY
#ifdef _STP_TEXTURE_SPLAT_BUILDER_H_

template<size_t... Is, class... Arg>
inline void SuperTerrainPlus::STPDiversity::STPTextureSplatBuilder::expandAddAltitudes(Sample sample, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get;

	//fold on comma operator
	(this->addAltitude(sample, 
		get<2 * Is + 0>(args), 
		get<2 * Is + 1>(args)
	), ...);
}

template<size_t ...Is, class ...Arg>
inline void SuperTerrainPlus::STPDiversity::STPTextureSplatBuilder::expandAddGradients(Sample sample, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get();

	(this->addGradient(sample, 
		get<5 * Is + 0>(args), 
		get<5 * Is + 1>(args), 
		get<5 * Is + 2>(args), 
		get<5 * Is + 3>(args), 
		get<5 * Is + 4>(args)
	), ...);
}

template<class... Arg>
void SuperTerrainPlus::STPDiversity::STPTextureSplatBuilder::addAltitudes(Sample sample, Arg&&... args) {
	//static assert is not required
	this->expandAddAltitudes(sample, std::make_index_sequence<sizeof...(Arg) / 2>, std::forward_as_tuple(args...));
}

template<class... Arg>
void SuperTerrainPlus::STPDiversity::STPTextureSplatBuilder::addGradients(Sample sample, Arg&&... args) {
	this->expandAddGradients(sample, std::make_index_sequence<sizeof...(Arg) / 5>, std::forward_as_tuple(args...));
}

#endif//_STP_TEXTURE_SPLAT_BUILDER_H_