#pragma once
#ifndef _STP_MESH_SETTINGS_HPP_
#define _STP_MESH_SETTINGS_HPP_

#include "STPSettings.hpp"

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
		 * @brief STPMeshSettings stores parameters for rendering terrain mesh
		*/
		struct STPMeshSettings: public STPSettings {
		public:

			/**
			 * @brief STPTessellationSettings controls the range of the tessellation levels, as well as the min and max distance
			 * where tessellation will become min and max
			*/
			struct STPTessellationSettings : public STPSettings {
			public:

				/**
				 * @brief Determine the maximum tessellation level when the distance falls beyond FurthestTessDistance
				*/
				float MaxTessLevel;

				/**
				 * @brief Determine the minumum tessellation level when the distance falls below NearestTessDistance
				*/
				float MinTessLevel;

				/**
				 * @brief Determine the maximum tessellation distance where tess level beyong will be clamped to MaxTessLevel
				*/
				float FurthestTessDistance;

				/**
				 * @brief Determine the minimum tessellation distance where tess level below will be clamped to MinTessLevel
				*/
				float NearestTessDistance;

				/**
				 * @brief Init STPTessellationSettings with defaults
				*/
				STPTessellationSettings() {
					this->MaxTessLevel = 0.0f;
					this->MinTessLevel = 0.0f;
					this->FurthestTessDistance = 0.0f;
					this->NearestTessDistance = 0.0f;
				}

				~STPTessellationSettings() = default;

				bool validate() override {
					return this->MaxTessLevel >= 0.0f
						&& this->MinTessLevel >= 0.0f
						&& this->FurthestTessDistance > 0.0f
						&& this->NearestTessDistance > 0.0f
						//range check
						&& this->MaxTessLevel >= this->MinTessLevel
						&& this->FurthestTessDistance >= this->NearestTessDistance;
				}

			};

			//Tessellation Control
			/**
			 * @brief Specify the height multiplier on the heightmap
			*/
			float Altitude;
			/**
			 * @brief Determine how far the mesh starts to decrease its LoD, (0, inf), in classic hermite interpolation, this factor will be 8.0f
			 * 2.0 is the default value, mesh will half its original LoD at 50% of tessllation distance
			*/
			float LoDShiftFactor;
			/**
			 * @brief Determine the range of tessellations, and the tessellation LoD will be clamped between min and max within a specified range of distances.
			*/
			STPTessellationSettings TessSettings;

			//Fragment Control

			/**
			 * @brief Init STPMeshSettings with defaults
			*/
			STPMeshSettings() {
				this->Altitude = 1.0f;
				this->LoDShiftFactor = 2.0f;
			}

			~STPMeshSettings() = default;

			bool validate() override {
				return this->Altitude > 0.0f
					&& this->LoDShiftFactor > 0.0f
					&& this->TessSettings.validate();
			}

		};

	}

}
#endif//_STP_MESH_SETTINGS_HPP_