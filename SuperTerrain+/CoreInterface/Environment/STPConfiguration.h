#pragma once
#ifndef _STP_CONFIGURATION_H_
#define _STP_CONFIGURATION_H_

#include <STPCoreDefine.h>
//Include all settings here
#include "STPChunkSetting.h"
#include "STPHeightfieldSetting.h"
#include "STPMeshSetting.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPEnvironment contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPEnvironment {

		/**
		 * @brief STPConfigurations stores configurations each settings of Super Terrain +
		*/
		class STP_API STPConfiguration : public STPSetting {
		private:

			STPChunkSetting ChunkSetting;
			STPHeightfieldSetting HeightfieldSetting;
			STPMeshSetting MeshSetting;

		public:

			/**
			 * @brief Init STPConfiguration with all settings set to their default
			*/
			STPConfiguration() = default;

			~STPConfiguration() = default;

			bool validate() const override;

			//------------------Get setting-------------------//

			/**
			 * @brief Get chunk setting
			 * @return Pointer to chunk setting
			*/
			STPChunkSetting& getChunkSetting();

			/**
			 * @brief Get heightfield setting
			 * @return Pointer to heightfield setting
			*/
			STPHeightfieldSetting& getHeightfieldSetting();

			/**
			 * @brief Get mesh setting
			 * @return Pointer to mesh setting
			*/
			STPMeshSetting& getMeshSetting();

		};

	}
}
#endif//_STP_CONFIGURATION_H_