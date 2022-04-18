#pragma once
#ifndef _STP_WATER_H_
#define _STP_WATER_H_

#include <SuperRealism+/STPRealismDefine.h>
//Scene
#include "../STPSceneObject.h"
//Dependent Terrain
#include "STPHeightfieldTerrain.h"
//Object
#include "../../Object/STPTexture.h"
#include "../../Object/STPBindlessTexture.h"

#include "../../Environment/STPWaterSetting.h"

//System
#include <unordered_map>
#include <optional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPWater draws a water plane on the procedural world and allows rendering of post effects such as reflection.
	 * Water is rendered as a plane and placed as defined by user with customisable altitude based on location on a biomemap.
	 * Due to its closed relationship to the heightfield terrain and shares common data with it, it should be used alongside with the terrain renderer.
	*/
	class STP_REALISM_API STPWater : public STPSceneObject::STPTransparentObject {
	private:

		//The water mesh shares the plane mesh with the terrain object so they can have the same size.
		const STPHeightfieldTerrain<false>& TerrainObject;

		//A texture to hold all water level data for each biome
		STPTexture WaterLevelTable;
		std::optional<STPBindlessTexture> WaterLevelTableHandle;

		STPProgramManager WaterAnimator;
		STPPipelineManager WaterRenderer;

		STPOpenGL::STPint WaveTimeLocation;

		//Wave animation logic
		double WavePhase;
		//The period is updated whenever water setting is set.
		double WavePeriod;

	public:

		//A lookup table define water height at different biomes.
		//The water level should use the same height metric as the terrain height value, and hence should be in range [0, 1].
		//If no water level is defined for a biome, the biome is assumed to have no water.
		typedef std::unordered_map<STPDiversity::Sample, float> STPBiomeWaterLevel;

		/**
		 * @brief Initialise a new water object.
		 * @param terrain The pointer to the terrain on top of which the water should be rendered.
		 * This terrain object should remain valid until the water object is destroyed.
		 * @param water_level A pointer to a dictionary for looking up water level per biome,
		*/
		STPWater(const STPHeightfieldTerrain<false>&, const STPBiomeWaterLevel&);

		STPWater(const STPWater&) = delete;

		STPWater(STPWater&&) = delete;

		STPWater& operator=(const STPWater&) = delete;

		STPWater& operator=(STPWater&&) = delete;

		~STPWater() = default;

		/**
		 * @brief Update water setting.
		 * @param water_setting The pointer to the water setting. Settings are copied.
		*/
		void setWater(const STPEnvironment::STPWaterSetting&);

		/**
		 * @brief Get wave period of the current water setting.
		 * @return The period of water wave. The returned value is valid only if the water setting has been set.
		*/
		double getWavePeriod() const;

		/**
		 * @brief Set the wave time ahead by a specified number, to animate the water wave.
		 * @param time The amount of time to advance.
		*/
		void advanceWaveTime(double);

		void render() const override;

	};

}
#endif//_STP_WATER_H_