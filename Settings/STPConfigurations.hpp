#pragma once
#ifndef _STP_CONFIGURATIONS_HPP_
#define _STP_CONFIGURATIONS_HPP_

//Include all settings here
#include "STPChunkSettings.hpp"
#include "STPHeightfieldSettings.hpp"
#include "STPMeshSettings.hpp"
#include "STPSimplexNoiseSettings.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSettings contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPSettings {

		/**
		 * @brief STPConfigurations stores configurations each settings of Super Terrain +
		*/
		class STPConfigurations : public STPSettings {
		private:

			STPChunkSettings ChunkSettings;
			STPHeightfieldSettings HeightfieldSettings;
			STPMeshSettings MeshSettings;
			STPSimplexNoiseSettings SimplexSettings;

		public:

			/**
			 * @brief Init STPConfigurations with all settings set to their default
			*/
			STPConfigurations() = default;

			~STPConfigurations() = default;

			bool validate() override {
				return this->ChunkSettings.validate()
					&& this->HeightfieldSettings.validate()
					&& this->MeshSettings.validate()
					&& this->SimplexSettings.validate();
			}

			//------------------Get settings-------------------//

			STPChunkSettings& getChunkSettings() {
				return this->ChunkSettings;
			}

			STPHeightfieldSettings& getHeightfieldSettings() {
				return this->HeightfieldSettings;
			}

			STPMeshSettings& getMeshSettings() {
				return this->MeshSettings;
			}

			STPSimplexNoiseSettings& getSimplexNoiseSettings() {
				return this->SimplexSettings;
			}

		};

	}
}
#endif//_STP_CONFIGURATIONS_HPP_