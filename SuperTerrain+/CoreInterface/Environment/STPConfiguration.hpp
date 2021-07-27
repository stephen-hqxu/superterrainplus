#pragma once
#ifndef _STP_CONFIGURATION_HPP_
#define _STP_CONFIGURATION_HPP_

//Include all settings here
#include "STPChunkSetting.hpp"
#include "STPHeightfieldSetting.hpp"
#include "STPMeshSetting.hpp"

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
		class STPConfiguration : public STPSetting {
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

			bool validate() const override {
				return this->ChunkSetting.validate()
					&& this->HeightfieldSetting.validate()
					&& this->MeshSetting.validate();
			}

			//------------------Get settings-------------------//

			STPChunkSetting& getChunkSetting() {
				return this->ChunkSetting;
			}

			STPHeightfieldSetting& getHeightfieldSetting() {
				return this->HeightfieldSetting;
			}

			STPMeshSetting& getMeshSetting() {
				return this->MeshSetting;
			}

		};

	}
}
#endif//_STP_CONFIGURATION_HPP_