#pragma once
#ifndef _STP_HEIGHTFIELD_PARA_LOADER_H_
#define _STP_HEIGHTFIELD_PARA_LOADER_H_

//INI loader
#include "SIMPLE/SIStorage.h"
//GLM
#include "glm/vec2.hpp"
//Settings
#include <SuperTerrain+/Environment/STPConfiguration.h>
#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>

namespace STPDemo {

	/**
	 * @brief STPTerrainParaLoader is a helper class to load terrain generation parameters from ini file to object that can be used by terrain generator 
	 * to create procedural terrain.
	*/
	namespace STPTerrainParaLoader {

		/**
		 * @brief Load the procedual 2d infinite terrain rendering parameters
		 * @param section The INI section that contains the 2d terrain rendering parameters
		 * @return The terrain rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPMeshSetting getProcedural2DINFRenderingParameter(const SIMPLE::SISection&);

		/**
		 * @brief Load the chunk settings and rendering parameters for procedural infinite 2d terrain
		 * @param section The INI section that contains the 2d terrian rendering parameters
		 * @return The chunk manager rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPChunkSetting getProcedural2DINFChunksParameter(const SIMPLE::SISection&);

		/**
		 * @brief Load the launch parameter for procedural infinite 2d terrain, stored in the ini file into STPHeightfieldLaunchPara
		 * @param section The INI section that contains the launch parameter
		 * @param slipRange The size of the free-slip range of the erosion
		 * @return The launch parameter, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPHeightfieldSetting getProcedural2DINFGeneratorParameter(const SIMPLE::SISection&, glm::uvec2);

		/**
		 * @brief Load the simplex noise 2d parameter, stored in the ini file into STPSimplexNoise2DPara
		 * @param section The INI section that contains the launch parameter
		 * @return The noise parameter,, if certain parameters are missing in the section, exception will eb thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting getSimplex2DNoiseParameter(const SIMPLE::SISection&);

		/**
		 * @brief Load all biome parameters into STPBiomeRegistry
		 * @param biomeini The INI file for the biome config
		*/
		void loadBiomeParameters(const SIMPLE::SIStorage&);

	};
}
#endif//_STP_HEIGHTFIELD_PARA_LOADER_H_