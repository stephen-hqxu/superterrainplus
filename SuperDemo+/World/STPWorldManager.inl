#pragma once
#ifdef _STP_WORLD_MANAGER_H_

template<class Base, class Ins, typename... Arg>
inline auto STPDemo::STPWorldManager::attach(Arg&&... arg) {
	//create an instace of attachment
	return std::unique_ptr<Base>(dynamic_cast<Base*>(new Ins(std::forward<Arg>(arg)...)));
	//else, impossible case to provide an invalid base class that sucessfully inited
}

template<class Fac, typename... Arg>
inline void STPDemo::STPWorldManager::attachBiomeFactory(Arg&&... arg) {
	using namespace SuperTerrainPlus::STPDiversity;
	//create an instance of biome factory
	//assign to our heightfield generator
	this->BiomeFactory = this->attach<STPBiomeFactory, Fac>(std::forward<Arg>(arg)...);
}

template<class Div, typename... Arg>
inline void STPDemo::STPWorldManager::attachDiversityGenerator(Arg&&... arg) {
	using namespace SuperTerrainPlus::STPCompute;
	//create an instance of diversity generator
	this->DiversityGenerator = this->attach<STPDiversityGenerator, Div>(std::forward<Arg>(arg)...);
}

template<class Tex, typename... Arg>
inline void STPDemo::STPWorldManager::attachTextureFactory(Arg&&... arg) {
	using namespace SuperTerrainPlus::STPDiversity;
	//create an instance of texture factory
	this->TextureFactory = this->attach<STPTextureFactory, Tex>(std::forward<Arg>(arg)...);
}

#endif//_STP_WORLD_MANAGER_H_