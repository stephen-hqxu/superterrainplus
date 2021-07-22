#pragma once
#ifdef _STP_WORLD_MANAGER_H_

template<class Fac, typename... Arg>
inline void SuperTerrainPlus::STPWorldManager::attachBiomeFactory(Arg&&... arg) {
	using namespace STPDiversity;
	//create an instance of biome factory
	//init a new biome factory
	STPBiomeFactory* factory = dynamic_cast<STPBiomeFactory*>(new Fac(std::forward<Arg>(arg)...));
	//assign to our heightfield generator
	this->BiomeFactory = std::unique_ptr<STPBiomeFactory>(factory);
}

#endif//_STP_WORLD_MANAGER_H_