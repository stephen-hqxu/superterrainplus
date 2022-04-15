#pragma once
#ifndef _STP_TESSELLATION_SETTING_H_
#define _STP_TESSELLATION_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/Environment/STPSetting.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPTessellationSettings controls the range of the tessellation levels, as well as the min and max distance
	 * where tessellation will become min and max
	*/
	struct STP_REALISM_API STPTessellationSetting : public STPSetting {
	public:

		/**
		 * @brief Determine the maximum tessellation level when the distance falls beyond FurthestTessDistance
		*/
		float MaxTessLevel;

		/**
		 * @brief Determine the minimum tessellation level when the distance falls below NearestTessDistance
		*/
		float MinTessLevel;

		/**
		 * @brief Determine the maximum tessellation distance where tess level beyond will be clamped to MaxTessLevel
		*/
		float FurthestTessDistance;

		/**
		 * @brief Determine the minimum tessellation distance where tess level below will be clamped to MinTessLevel
		*/
		float NearestTessDistance;

		/**
		 * @brief Init STPTessellationSettings with defaults
		*/
		STPTessellationSetting();

		~STPTessellationSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_TESSELLATION_SETTING_H_