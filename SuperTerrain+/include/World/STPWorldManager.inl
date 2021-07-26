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

template<class Div, typename... Arg>
inline void SuperTerrainPlus::STPWorldManager::attachDiversityGenerator(Arg&&... arg) {
	using namespace STPCompute;
	//create an instance of diversity generator
	STPDiversityGenerator* diversity = dynamic_cast<STPDiversityGenerator*>(new Div(std::forward<Arg>(arg)...));
	//assign
	this->DiversityGenerator = std::unique_ptr<STPDiversityGenerator>(diversity);
}

#endif//_STP_WORLD_MANAGER_H_