#pragma once
#ifndef _STP_CONFIGURATION_H_
#define _STP_CONFIGURATION_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Include all settings here
#include "STPChunkSetting.h"
#include "STPHeightfieldSetting.h"
#include "STPMeshSetting.h"

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPConfigurations stores configurations each settings of Super Terrain +
	*/
	class STP_API STPConfiguration : public STPSetting {
	public:

		STPChunkSetting ChunkSetting;
		STPHeightfieldSetting HeightfieldSetting;
		STPMeshSetting MeshSetting;

		//STPHeightfieldSetting is non-copiable

		/**
		 * @brief Init STPConfiguration with all settings set to their default
		*/
		STPConfiguration() = default;

		STPConfiguration(const STPConfiguration&) = delete;

		STPConfiguration(STPConfiguration&&) noexcept = default;

		STPConfiguration& operator=(const STPConfiguration&) = delete;

		STPConfiguration& operator=(STPConfiguration&&) noexcept = default;

		~STPConfiguration() = default;

		bool validate() const override;

	};

}
#endif//_STP_CONFIGURATION_H_