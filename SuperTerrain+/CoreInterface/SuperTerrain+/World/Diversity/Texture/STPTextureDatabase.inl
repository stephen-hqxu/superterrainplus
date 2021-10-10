//PLEASE DO NOT INCLUDE THIS FILE SEPARATELY
#ifdef _STP_TEXTURE_DATABASE_H_

template<size_t... Is, class... Arg>
inline auto SuperTerrainPlus::STPDiversity::STPTextureDatabase::expandAddTextures(STPTextureID texture_id, std::index_sequence<Is...>, std::tuple<Arg...> args) {
	using std::get;
	
	return std::array<bool, sizeof...(Is)>{
		this->addTextureData(texture_id,
			get<3 * Is + 0>(args),
			get<3 * Is + 1>(args),
			get<3 * Is + 2>(args)
		)...};
}

template<class... Arg>
auto SuperTerrainPlus::STPDiversity::STPTextureDatabase::addTextureDatas(STPTextureID texture_id, Arg&&... args) {
	//no need to check for parameter pack size, compiler will throw an error if arguments are not multiple of 3, or there's no argument, etc.
	//because addTexture() function requires the exact signature

	//convert parameter packs into a tuple so we can index it
	return this->expandAddTextures(texture_id, std::make_index_sequence<sizeof...(Arg) / 3ull>(), std::forward_as_tuple(args...));
}

#endif//_STP_TEXTURE_DATABASE_H_