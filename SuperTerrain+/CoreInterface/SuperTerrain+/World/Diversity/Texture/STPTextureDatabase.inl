//PLEASE DO NOT INCLUDE THIS FILE SEPARATELY
#ifdef _STP_TEXTURE_DATABASE_H_

template<size_t... Is, class... Arg>
inline void SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPTextureSplatBuilder::expandAddAltitudes(Sample sample, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get;

	//fold on comma operator
	(this->addAltitude(sample,
		get<2 * Is + 0>(args),
		get<2 * Is + 1>(args)
	), ...);
}

template<size_t... Is, class... Arg>
inline void SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPTextureSplatBuilder::expandAddGradients(Sample sample, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get;

	(this->addGradient(sample,
		get<5 * Is + 0>(args),
		get<5 * Is + 1>(args),
		get<5 * Is + 2>(args),
		get<5 * Is + 3>(args),
		get<5 * Is + 4>(args)
	), ...);
}

template<class... Arg>
void SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPTextureSplatBuilder::addAltitudes(Sample sample, Arg&&... args) {
	//static assert is not required
	this->expandAddAltitudes(sample, std::make_index_sequence<sizeof...(Arg) / 2ull>(), std::forward_as_tuple(args...));
}

template<class... Arg>
void SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPTextureSplatBuilder::addGradients(Sample sample, Arg&&... args) {
	this->expandAddGradients(sample, std::make_index_sequence<sizeof...(Arg) / 5ull>(), std::forward_as_tuple(args...));
}

template<size_t... Is, class... Arg>
inline void SuperTerrainPlus::STPDiversity::STPTextureDatabase::expandAddMaps(STPTextureInformation::STPTextureID texture_id, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get;
	
	(this->addMap(texture_id,
		get<3 * Is + 0>(args),
		get<3 * Is + 1>(args),
		get<3 * Is + 2>(args)
	), ...);
}

template<class... Arg>
void SuperTerrainPlus::STPDiversity::STPTextureDatabase::addMaps(STPTextureInformation::STPTextureID texture_id, Arg&&... args) {
	//no need to check for parameter pack size, compiler will throw an error if arguments are not multiple of 3, or there's no argument, etc.
	//because addTexture() function requires the exact signature

	//convert parameter packs into a tuple so we can index it
	this->expandAddMaps(texture_id, std::make_index_sequence<sizeof...(Arg) / 3ull>(), std::forward_as_tuple(args...));
}

#endif//_STP_TEXTURE_DATABASE_H_