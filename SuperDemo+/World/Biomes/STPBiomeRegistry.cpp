#include "STPBiomeRegistry.h"

using SuperTerrainPlus::STPSample_t;
using namespace STPDemo;
namespace Reg = STPBiomeRegistry;

/* -------------------------------- Definition of Biome Variables -------------------------- */
std::unordered_map<STPSample_t, const STPBiome*> Reg::Registry;

#define DEF_BIOME(NAME) STPBiome Reg::NAME
DEF_BIOME(Ocean);
DEF_BIOME(DeepOcean);

DEF_BIOME(WarmOcean);
DEF_BIOME(LukewarmOcean);
DEF_BIOME(ColdOcean);
DEF_BIOME(FrozenOcean);
DEF_BIOME(DeepWarmOcean);
DEF_BIOME(DeepLukewarmOcean);
DEF_BIOME(DeepColdOcean);
DEF_BIOME(DeepFrozenOcean);

DEF_BIOME(Plains);
DEF_BIOME(Desert);
DEF_BIOME(DesertHills);
DEF_BIOME(Mountain);
DEF_BIOME(WoodedMountain);
DEF_BIOME(SnowyMountain);
DEF_BIOME(Forest);
DEF_BIOME(ForestHills);
DEF_BIOME(Taiga);
DEF_BIOME(TaigaHills);
DEF_BIOME(SnowyTaiga);
DEF_BIOME(SnowyTundra);

DEF_BIOME(River);
DEF_BIOME(FrozenRiver);
DEF_BIOME(Beach);
DEF_BIOME(SnowyBeach);
DEF_BIOME(StoneShore);

DEF_BIOME(Jungle);
DEF_BIOME(JungleHills);
DEF_BIOME(Savannah);
DEF_BIOME(SavannahPlateau);
DEF_BIOME(Swamp);
DEF_BIOME(SwampHills);
DEF_BIOME(Badlands);
DEF_BIOME(BadlandsPlateau);

/* ---------------------------------- Biome Registry Function ------------------------ */

void Reg::registerBiomes() {
	static bool initialised = false;
	if (initialised) {
		//do not re-initialise those biomes
		return;
	}

	//add all biomes to registry
	static const auto reg_insert = [](const STPBiome& biome) -> void {
		Reg::Registry.emplace(biome.ID, &biome);
		return;
	};
	//Oceans
	reg_insert(STPBiomeRegistry::Ocean);
	reg_insert(STPBiomeRegistry::DeepOcean);
	reg_insert(STPBiomeRegistry::WarmOcean);
	reg_insert(STPBiomeRegistry::LukewarmOcean);
	reg_insert(STPBiomeRegistry::ColdOcean);
	reg_insert(STPBiomeRegistry::FrozenOcean);
	reg_insert(STPBiomeRegistry::DeepWarmOcean);
	reg_insert(STPBiomeRegistry::DeepLukewarmOcean);
	reg_insert(STPBiomeRegistry::DeepColdOcean);
	reg_insert(STPBiomeRegistry::DeepFrozenOcean);
	//Rivers
	reg_insert(STPBiomeRegistry::River);
	reg_insert(STPBiomeRegistry::FrozenRiver);
	//Lands
	reg_insert(STPBiomeRegistry::Plains);
	reg_insert(STPBiomeRegistry::Desert);
	reg_insert(STPBiomeRegistry::Mountain);
	reg_insert(STPBiomeRegistry::Forest);
	reg_insert(STPBiomeRegistry::Taiga);
	reg_insert(STPBiomeRegistry::SnowyTaiga);
	reg_insert(STPBiomeRegistry::SnowyTundra);
	reg_insert(STPBiomeRegistry::Jungle);
	reg_insert(STPBiomeRegistry::Savannah);
	reg_insert(STPBiomeRegistry::Swamp);
	reg_insert(STPBiomeRegistry::Badlands);
	//Hills
	reg_insert(STPBiomeRegistry::DesertHills);
	reg_insert(STPBiomeRegistry::TaigaHills);
	reg_insert(STPBiomeRegistry::WoodedMountain);
	reg_insert(STPBiomeRegistry::SnowyMountain);
	reg_insert(STPBiomeRegistry::ForestHills);
	reg_insert(STPBiomeRegistry::JungleHills);
	reg_insert(STPBiomeRegistry::SavannahPlateau);
	reg_insert(STPBiomeRegistry::SwampHills);
	reg_insert(STPBiomeRegistry::BadlandsPlateau);
	//Edges and Shores
	reg_insert(STPBiomeRegistry::Beach);
	reg_insert(STPBiomeRegistry::SnowyBeach);
	reg_insert(STPBiomeRegistry::StoneShore);

	initialised = true;
}

bool Reg::isShallowOcean(const STPSample_t val) noexcept {
	return val == STPBiomeRegistry::Ocean.ID || val == STPBiomeRegistry::FrozenOcean.ID
		|| val == STPBiomeRegistry::WarmOcean.ID || val == STPBiomeRegistry::LukewarmOcean.ID
		|| val == STPBiomeRegistry::ColdOcean.ID;
}

bool Reg::isOcean(const STPSample_t val) noexcept {
	return STPBiomeRegistry::isShallowOcean(val) || val == STPBiomeRegistry::DeepOcean.ID
		|| val == STPBiomeRegistry::DeepWarmOcean.ID || val == STPBiomeRegistry::DeepLukewarmOcean.ID
		|| val == STPBiomeRegistry::DeepColdOcean.ID || val == STPBiomeRegistry::DeepFrozenOcean.ID;
}

bool Reg::isRiver(const STPSample_t val) noexcept {
	return val == STPBiomeRegistry::River.ID || val == STPBiomeRegistry::FrozenRiver.ID;
}

Reg::STPPrecipitationType Reg::getPrecipitationType(const STPSample_t val) {
	const STPBiome& biome = *STPBiomeRegistry::Registry[val];

	//we check for precipitation first, some biome like taiga, even it's cold but it's dry so it won't snow nor rain
	//of course we could have a more elegant model to determine the precipitation type, but let's just keep it simple
	if (biome.Precipitation < 1.0f) {
		//desert and savannah usually has precipitation less than 1.0
		return STPPrecipitationType::NONE;
	}

	if (biome.Temperature < 1.0f) {
		//snowy biome has temp less than 1.0
		return STPPrecipitationType::SNOW;
	}

	return STPPrecipitationType::RAIN;
}

STPSample_t Reg::CAS(const STPSample_t comparator, const STPSample_t comparable, const STPSample_t fallback) noexcept {
	return comparator == comparable ? comparable : fallback;
}