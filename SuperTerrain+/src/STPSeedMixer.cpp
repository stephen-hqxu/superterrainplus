#include <World/Biome/STPSeedMixer.h>

using namespace SuperTerrainPlus::STPDiversity;

STPSeedMixer::STPSeedMixer() {

}

STPSeedMixer::~STPSeedMixer() {

}

Seed STPSeedMixer::mixSeed(Seed s, long long fac) {
	//TODO: Mix seed based on any algorithm you like, feel free to change the implementation
	s *= s * 6364136223846793005ull + 1442695040888963407ull;
	s += fac;
	return s;
}