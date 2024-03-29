#pragma once
//defined by source compiler only
#ifndef STP_IMPLEMENTATION
#error __FILE__ is an implementation of hydraulic erosion but shall not be used in external environment
#endif

#ifndef _STP_RAINDROP_CUH_
#define _STP_RAINDROP_CUH_

#include "../World/STPWorldMapPixelFormat.hpp"
//Settings
#include "../Environment/STPRainDropSetting.h"
#include "../World/Chunk/STPErosionBrush.hpp"

//CUDA Runtime
#include <cuda_runtime.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPRainDrop class spawn a water droplet at a random location on the map and perform hydraulic erosion for 2D heightmap.
	 * The generative system that can capture a number of geographical phenomena, including:
	 * - River and Stream Migration
	 * - Natural Waterfalls
	 * - Canyon Formation
	 * - Swelling and Floodplains
	 * The hydrology and terrain systems must thus both be dynamic and strongly coupled. Particle-based hydraulic erosion already has the core aspects needed for this:
	 * - Terrain affects the movement of water
	 * - Erosion and sedimentation affect the terrain
	 * This system effectively models the erosive effects of rain, but fails to capture a number of other effects:
	 * - Water behaves differently in a moving stream
	 * - Water behaves differently in a stagnant pool
	*/
	class STPRainDrop {
	private:

		//The current position of the raindrop
		glm::vec2 Position;
		//Direction of water flow
		glm::vec2 Direction;
		//Flowing speed
		float Speed;
		//Water volume
		float Volume;
		//The amount of sediment it carries
		float Sediment = 0.0f;

		const glm::uvec2 Dimension;

		/**
		 * @brief Calculate the current height of the water drop and the direction of acceleration
		 * @param map The floating point heightmap with free slip configurations
		 * @return Height and Gradients, will be defined in vec3 as (height, gradientX, gradientY);
		*/
		__device__ glm::vec3 calcHeightGradients(const STPHeightFloat_t*) const;

	public:

		/**
		 * @brief It starts raining! Let's produce a rain drop!
		 * @param position - The position of the rain drop. It's recommend to randomly generate a position to simulate natural randomness.
		 * @param WaterVolume - The initial water volume.
		 * @param MovementSpeed - The initial speed of the droplet.
		 * @param dimension The dimension of the heightmap.
		*/
		__device__ STPRainDrop(glm::vec2, float, float, glm::uvec2);

		~STPRainDrop() = default;

		/**
		 * @brief Performing hydraulic erosion algorithm to descend the raindrop downhill once, 
		 * water drop will bring sediment but lose water each time this method is called.
		 * @param map - The floating point heightmap with free slip configurations.
		 * @param settings - The raindrop settings for erosion.
		 * @param brush - The generated erosion brush table.
		*/
		__device__ void operator()(STPHeightFloat_t*, const STPEnvironment::STPRainDropSetting&, const STPErosionBrush&);

	};

}
#endif//_STP_RAINDROP_CUH_