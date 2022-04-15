#pragma once
#ifndef _STP_WATER_H_
#define _STP_WATER_H_

#include <SuperRealism+/STPRealismDefine.h>
//Scene
#include "../STPSceneObject.h"
//Dependent Terrain
#include "STPHeightfieldTerrain.h"

#include "../../Environment/STPWaterSetting.h"

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

		//fragment shader only, all other shader stages are inherited from the terrain object.
		STPProgramManager WaterAnimator;
		STPPipelineManager WaterRenderer;

	public:

		/**
		 * @brief Initialise a new water object.
		 * @param terrain The pointer to the terrain on top of which the water should be rendered.
		 * This terrain object should remain valid until the water object is destroyed.
		*/
		STPWater(const STPHeightfieldTerrain<false>&);

		STPWater(const STPWater&) = delete;

		STPWater(STPWater&&) = delete;

		STPWater& operator=(const STPWater&) = delete;

		STPWater& operator=(STPWater&&) = delete;

		~STPWater() = default;

		void setWater(const STPEnvironment::STPWaterSetting&);

		void waveTime(unsigned long long);

		void render() const override;

	};

}
#endif//_STP_WATER_H_