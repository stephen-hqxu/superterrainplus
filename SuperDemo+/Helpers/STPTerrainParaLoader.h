#pragma once
#ifndef _STP_HEIGHTFIELD_PARA_LOADER_H_
#define _STP_HEIGHTFIELD_PARA_LOADER_H_

//INI
#include <SuperAlgorithm+/Parser/INI/STPINIStorage.hpp>
//GLM
#include <glm/vec2.hpp>
//Settings
#include <SuperTerrain+/Environment/STPConfiguration.h>
#include <SuperAlgorithm+/STPSimplexNoiseSetting.h>
#include <SuperRealism+/Environment/STPMeshSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>
#include <SuperRealism+/Environment/STPAtmosphereSetting.h>
#include <SuperRealism+/Environment/STPOcclusionKernelSetting.h>
#include <SuperRealism+/Environment/STPWaterSetting.h>
#include <SuperRealism+/Environment/STPBidirectionalScatteringSetting.h>

#include <utility>

namespace STPDemo {

	/**
	 * @brief STPTerrainParaLoader is a helper class to load terrain generation parameters from INI file to object that can be used by terrain generator 
	 * to create procedural terrain.
	*/
	namespace STPTerrainParaLoader {

		/**
		 * @brief Load the terrain rendering settings.
		 * @param section The INI section that contains the 2d terrain rendering parameters
		 * @return The terrain rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPMeshSetting getRenderingSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load the chunk and rendering settings for procedural terrain
		 * @param section The INI section that contains the 2d terrain rendering parameters
		 * @return The chunk manager rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPChunkSetting getChunkSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load the launch settings for terrain.
		 * @param section The INI section that contains the launch parameter
		 * @param slipRange The size of the free-slip range of the erosion
		 * @return The launch parameter, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPHeightfieldSetting getGeneratorSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&, glm::uvec2);

		/**
		 * @brief Load the simplex noise setting.
		 * @param section The INI section that contains the launch parameter
		 * @return The noise parameter, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting getSimplexSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load the settings for procedural sky rendering.
		 * @param section The INI section that contains all sky setting.
		 * @return Setting for sun and atmosphere.
		*/
		std::pair<SuperTerrainPlus::STPEnvironment::STPSunSetting, SuperTerrainPlus::STPEnvironment::STPAtmosphereSetting>
			getSkySetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load the settings for ambient occlusion.
		 * @param section The INI section that contains the AO setting.
		 * @return The AO setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPOcclusionKernelSetting getAOSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load the settings for water rendering.
		 * @param section The INI section contains water setting.
		 * @param altitude The altitude of the terrain.
		 * @return The water setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPWaterSetting getWaterSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&, float);

		/**
		 * @brief Load the settings for BSDF rendering.
		 * @param section The INI section contains BSDF settings.
		 * @return The BSDF setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPBidirectionalScatteringSetting getBSDFSetting(const SuperTerrainPlus::STPAlgorithm::STPINISectionView&);

		/**
		 * @brief Load all biome parameters into STPBiomeRegistry
		 * @param biomeini The INI file for the biome config
		*/
		void loadBiomeParameters(const SuperTerrainPlus::STPAlgorithm::STPINIStorageView&);

	};
}
#endif//_STP_HEIGHTFIELD_PARA_LOADER_H_