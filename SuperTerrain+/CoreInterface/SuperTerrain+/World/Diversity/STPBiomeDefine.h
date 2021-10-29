#pragma once
#ifndef _STP_BIOME_DEF_H_
#define _STP_BIOME_DEF_H_

namespace SuperTerrainPlus::STPDiversity {
	//Sample of the layer, it can be interpreted as biome id, or temp, or climate, or anything based on implementation
	typedef unsigned short Sample;
	//A seed is a random factor that is used to generate a random sequence, the same seed will guarantee the same generated sequence
	typedef unsigned long long Seed;
}
#endif//_STP_BIOME_DEF_H_