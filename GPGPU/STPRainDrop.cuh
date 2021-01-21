#pragma once
#ifndef _STP_RAINDROP_CUH_
#define _STP_RAINDROP_CUH_

//CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//ADT
#include <list>
//Settings
#include "../Settings/STPRainDropSettings.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {

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
			float2 raindrop_pos;
			//Direction of water flow
			float2 raindrop_dir;
			//Flowing speed
			float speed;
			//Flowing velocity
			float2 velocity;
			//Water volume
			float volume;
			//The amount of sediment it carries
			float sediment = 0.0f;

			/**
			 * @brief Calculate the current height of the water drop and the direction of acceleration
			 * @param heightmap - The heightmap
			 * @param mapSize - The size of the heightmap
			 * @param position - The current position of the water drop
			 * @return Height and Gradients, will be defined in vec3 as (height, gradientX, gradientY);
			*/
			__device__ static float3 calcHeightGradients(float* const, uint2, float2);

		public:

			/**
			 * @brief It starts raining! Let's produce a rain drop!
			 * @param position - The position of the rain drop. It's recommend to randomly generate a position to simulate natrual randomness
			 * @param WaterVolume - The initial water volume
			 * @param MovementSpeed - The initial speed of the droplet
			*/
			__device__ STPRainDrop(float2, float, float);

			__device__ ~STPRainDrop();

			/**
			 * @brief Get the current water content in the droplet
			 * @return The current volume in the water droplet
			*/
			__device__ float getCurrentVolume();

			/**
			 * @brief Performing hydraulic erosion algorithm to descend the raindrop downhill once, water drop will bring sediment but lose water each time this method is called
			 * @param mapSize - The size of three maps, all three maps must have the same size
			 * @param heightmap - The heightmap data, should be precomputed
			 * @param settings - The raindrop settings for erosion
			*/
			__device__ void Erode(STPSettings::STPRainDropSettings* const, uint2, float*);

		};

	}
}
#endif//_STP_RAINDROP_CUH_