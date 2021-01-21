#pragma once
#ifndef _STP_HEIGHTFIELD_SETTINGS_HPP_
#define _STP_HEIGHTFIELD_SETTINGS_HPP_

#include "STPRainDropSettings.hpp"

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
		 * @brief STPHeightfieldSettings stores all heightfield parameters for compute launch
		*/
		struct STPHeightfieldSettings: public STPRainDropSettings {
		public:

			//Heightmap Parameters
			//Determine the zooming of the noise map
			float Scale;
			//Control how many heightmap will be conbined
			unsigned int Octave;
			//Control how the amplitude will be changed in each octave. Range (0,1)
			float Persistence;
			//Control how the frequency will be changed in each octave.
			float Lacunarity;

			//Normalmap Parameters
			//Control the strength of z component of the normal map, the greater, the more the normal pointing towards the surface
			float Strength;

			//Hydraulic Erosion Parameters are inherited from super class

			/**
			 * @brief Init STPHeightfieldSettings with defaults
			*/
			STPHeightfieldSettings() : STPRainDropSettings() {
				this->Scale = 1.0f;
				this->Octave = 1;
				this->Persistence = 1.0f;
				this->Lacunarity = 1.0f;
				this->Strength = 1.0f;
			}

			//Copy the source heightfield setting to the new instance
			STPHeightfieldSettings(const STPHeightfieldSettings& obj) : STPRainDropSettings(obj) {
				this->Scale = obj.Scale;
				this->Octave = obj.Octave;
				this->Persistence = obj.Persistence;
				this->Lacunarity = obj.Lacunarity;
				this->Strength = obj.Strength;
			}

			//Move the source setting to the new instance
			STPHeightfieldSettings(STPHeightfieldSettings&& obj) noexcept : STPRainDropSettings(std::forward<STPRainDropSettings>(obj)) {
				this->Scale = std::exchange(obj.Scale, 0.0f);
				this->Octave = std::exchange(obj.Octave, 0u);
				this->Persistence = std::exchange(obj.Persistence, 0.0f);
				this->Lacunarity = std::exchange(obj.Lacunarity, 0.0f);
				this->Strength = std::exchange(obj.Strength, 0.0f);
			}

			~STPHeightfieldSettings() = default;

			//Copy the source setting to current instance
			STPHeightfieldSettings& operator=(const STPHeightfieldSettings& obj) {
				//copy the base class
				STPRainDropSettings::operator=(obj);

				this->Scale = obj.Scale;
				this->Octave = obj.Octave;
				this->Persistence = obj.Persistence;
				this->Lacunarity = obj.Lacunarity;
				this->Strength = obj.Strength;

				return *this;
			}

			//Move source setting to current instance
			STPHeightfieldSettings& operator=(STPHeightfieldSettings&& obj) noexcept {
				//move the base class
				STPRainDropSettings::operator=(std::forward<STPRainDropSettings>(obj));

				this->Scale = std::exchange(obj.Scale, 0.0f);
				this->Octave = std::exchange(obj.Octave, 0u);
				this->Persistence = std::exchange(obj.Persistence, 0.0f);
				this->Lacunarity = std::exchange(obj.Lacunarity, 0.0f);
				this->Strength = std::exchange(obj.Strength, 0.0f);

				return *this;
			}

			bool validate() const override {
				static auto checkRange = []__host__(float value, float lower, float upper) -> bool {
					return value >= lower && value <= upper;
				};
				//check the raindrop parameter plus also heightmap parameter
				return this->STPRainDropSettings::validate()
					&& this->Scale > 0.0f
					&& this->Octave != 0u
					&& checkRange(this->Persistence, 0.0f, 1.0f)
					&& this->Lacunarity >= 1.0f
					&& this->Strength > 0.0f;
			}
		};

	}

}
#endif//_STP_HEIGHTFIELD_SETTINGS_HPP_