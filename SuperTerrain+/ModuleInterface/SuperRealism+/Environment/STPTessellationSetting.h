#pragma once
#ifndef _STP_TESSELLATION_SETTING_H_
#define _STP_TESSELLATION_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPTessellationSettings controls the range of the tessellation levels, as well as the min and max distance
	 * where tessellation will become min and max
	*/
	struct STP_REALISM_API STPTessellationSetting {
	public:

		//Determine the maximum tessellation level when the distance falls beyond FurthestTessDistance.
		float MaxTessLevel;
		//Determine the minimum tessellation level when the distance falls below NearestTessDistance.
		float MinTessLevel;
		//Determine the maximum tessellation distance ratio relative to max viewing distance where tess level beyond will be clamped to MaxTessLevel.
		float FurthestTessDistance;
		//Determine the minimum tessellation distance ratio relative to max viewing distance where tess level below will be clamped to MinTessLevel.
		float NearestTessDistance;

		void validate() const;

	};

}
#endif//_STP_TESSELLATION_SETTING_H_