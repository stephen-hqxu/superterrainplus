#pragma once
#ifndef _STP_HEIGHTFIELD_PARA_LOADER_H_
#define _STP_HEIGHTFIELD_PARA_LOADER_H_

//System
#include <string>
//INI loader
#include "SIMPLE1.0/SIStorage.h"
//GLM
#include "glm/vec2.hpp"
//Settings
#include <Settings/STPConfigurations.hpp>
#include <SuperAlgorithm+/STPSimplexNoiseSettings.hpp>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPTerrainParaLoader is a helper class to load terrain generation parameters from ini file to object that can be used by terrain generator 
	 * to create procedural terrain.
	*/
	class STPTerrainParaLoader {
	private:

		/**
		 * @brief static class, should not be instantiated
		*/
		STPTerrainParaLoader() = default;

		~STPTerrainParaLoader() = default;

		static const std::string Procedural2DINFRenderingVariables[6];
		static const std::string Procedural2DINFChunksVariables[14];
		static const std::string Procedural2DINFGeneratorVariables[15];
		static const std::string Simplex2DNoiseVariables[3];
		static const std::string BiomeVariables[10];

	public:

		/**
		 * @brief Load the procedual 2d infinite terrain rendering parameters
		 * @param section The INI section that contains the 2d terrain rendering parameters
		 * @return The terrain rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		static STPSettings::STPMeshSettings getProcedural2DINFRenderingParameters(const SIMPLE::SISection&);

		/**
		 * @brief Load the chunk settings and rendering parameters for procedural infinite 2d terrain
		 * @param section The INI section that contains the 2d terrian rendering parameters
		 * @return The chunk manager rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		static STPSettings::STPChunkSettings getProcedural2DINFChunksParameters(const SIMPLE::SISection&);

		/**
		 * @brief Load the launch parameter for procedural infinite 2d terrain, stored in the ini file into STPHeightfieldLaunchPara
		 * @param section The INI section that contains the launch parameter
		 * @param slipRange The size of the free-slip range of the erosion
		 * @return The launch parameter, if certain parameters are missing in the section, exception will be thrown
		*/
		static STPSettings::STPHeightfieldSettings getProcedural2DINFGeneratorParameters(const SIMPLE::SISection&, glm::uvec2);

		/**
		 * @brief Load the simplex noise 2d parameter, stored in the ini file into STPSimplexNoise2DPara
		 * @param section The INI section that contains the launch parameter
		 * @return The noise parameter,, if certain parameters are missing in the section, exception will eb thrown
		*/
		static STPSettings::STPSimplexNoiseSettings getSimplex2DNoiseParameters(const SIMPLE::SISection&);

		/**
		 * @brief Load all biome parameters into STPBiomeRegistry
		 * @param biomeini The INI file for the biome config
		*/
		static void loadBiomeParameters(const SIMPLE::SIStorage&);

	};
}
#endif//_STP_HEIGHTFIELD_PARA_LOADER_H_