#pragma once
#ifndef _STP_RAIN_DROP_SETTING_H_
#define _STP_RAIN_DROP_SETTING_H_

#include <SuperTerrain+/STPCoreDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPRainDropSetting stores all calculation parameters for the rain drops.
	 * Copy and move operation on this class is unsafe if device memory has been made available, and only shallow copy is performed.
	 * To make device memory available across all copies, only call makeAvailable() function on the object that has been copied.
	*/
	struct STP_API STPRainDropSetting {
	public:

		//The number of raindrop presented to perform hydraulic erosion
		unsigned int RainDropCount;
		//At zero, water will instantly change direction to flow downhill. At one, water will never change direction. Ranged [0,1]
		float Inertia;
		//Multiplier for how much sediment a droplet can carry.
		float SedimentCapacityFactor;
		//Used to prevent carry capacity getting too close to zero on flatten the entire terrain.
		float minSedimentCapacity;
		//The starting volume of the water droplet
		float initWaterVolume;
		//Used to check when to end the droplet's life time. If the current volume falls below 
		float minWaterVolume;
		//The greater the friction, the quicker the droplet will slow down. Ranged [0,1]
		float Friction;
		//The starting speed of the water droplet
		float initSpeed;
		//How fast a droplet will be filled with sediment. Ranged [0,1]
		float ErodeSpeed;
		//Limit the speed of sediment drop if exceeds the minSedimentCapacity. Ranged [0,1]
		float DepositSpeed;
		//Control how fast the water will evaporate in the droplet. Ranged [0,1]
		float EvaporateSpeed;
		//Control how fast water droplet descends.
		float Gravity;
		//Determine the radius of the droplet that can brush out sediment
		//Specify the radius of the brush. Determines the radius in which sediment is taken from the rock layer.
		//The smaller radius is, the deeper and more distinct the ravines will be.
		unsigned int ErosionBrushRadius;

		void validate() const;

	};

}
#endif//_STP_RAIN_DROP_SETTING_H_