#pragma once
#ifndef _STP_HEIGHTFIELD_PARA_LOADER_H_
#define _STP_HEIGHTFIELD_PARA_LOADER_H_

#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

//INI
#include <SuperAlgorithm+Host/Parser/STPINIData.hpp>

//Settings
#include <SuperTerrain+/Environment/STPChunkSetting.h>
#include <SuperTerrain+/Environment/STPRainDropSetting.h>
#include <SuperAlgorithm+Host/STPSimplexNoiseSetting.h>
#include <SuperRealism+/Environment/STPMeshSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>
#include <SuperRealism+/Environment/STPAtmosphereSetting.h>
#include <SuperRealism+/Environment/STPOcclusionKernelSetting.h>
#include <SuperRealism+/Environment/STPWaterSetting.h>
#include <SuperRealism+/Environment/STPBidirectionalScatteringSetting.h>
#include <SuperRealism+/Environment/STPStarfieldSetting.h>
#include <SuperRealism+/Environment/STPAuroraSetting.h>

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
		SuperTerrainPlus::STPEnvironment::STPMeshSetting getRenderingSetting(const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&);

		/**
		 * @brief Load the chunk and rendering settings for procedural terrain
		 * @param section The INI section that contains the 2d terrain rendering parameters
		 * @return The chunk manager rendering parameters, if certain parameters are missing in the section, exception will be thrown
		*/
		SuperTerrainPlus::STPEnvironment::STPChunkSetting getChunkSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&);

		/**
		 * @brief Load the settings for terrain erosion.
		 * @param section The INI section that contains the launch parameter.
		 * @param generator_seed The generator seed.
		 * @return The heightfield erosion parameters, if certain parameters are missing in the section, exception will be thrown.
		*/
		SuperTerrainPlus::STPEnvironment::STPRainDropSetting getRainDropSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&, SuperTerrainPlus::STPSeed_t);

		/**
		 * @brief Load the simplex noise setting..
		 * @param section The INI section that contains the launch parameter.
		 * @param simplex_seed The simplex noise seed.
		 * @return The noise parameter, if certain parameters are missing in the section, exception will be thrown.
		*/
		SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting getSimplexSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&, SuperTerrainPlus::STPSeed_t);

		/**
		 * @brief Load the settings for procedural sky rendering.
		 * @param section The INI section that contains all sky setting.
		 * @return Setting for sun and atmosphere.
		*/
		std::pair<SuperTerrainPlus::STPEnvironment::STPSunSetting, SuperTerrainPlus::STPEnvironment::STPAtmosphereSetting>
			getSkySetting(const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&);

		/**
		 * @brief Load the settings for procedural starfield rendering.
		 * @param section The INI section contains the star settings.
		 * @param star_seed The seed for starfield.
		 * @return Setting for starfield.
		*/
		SuperTerrainPlus::STPEnvironment::STPStarfieldSetting getStarfieldSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&, SuperTerrainPlus::STPSeed_t);

		/**
		 * @brief Load the settings for procedural aurora rendering.
		 * @param section The INI section.
		 * @return Aurora setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPAuroraSetting getAuroraSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&);

		/**
		 * @brief Load the settings for ambient occlusion.
		 * @param section The INI section that contains the AO setting.
		 * @param ao_seed The ambient occlusion seed.
		 * @return The AO setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPOcclusionKernelSetting getAOSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&, SuperTerrainPlus::STPSeed_t);

		/**
		 * @brief Load the settings for water rendering.
		 * @param section The INI section contains water setting.
		 * @param altitude The altitude of the terrain.
		 * @return The water setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPWaterSetting getWaterSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&, float);

		/**
		 * @brief Load the settings for BSDF rendering.
		 * @param section The INI section contains BSDF settings.
		 * @return The BSDF setting.
		*/
		SuperTerrainPlus::STPEnvironment::STPBidirectionalScatteringSetting getBSDFSetting(
			const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINISectionView&);

		/**
		 * @brief Load all biome parameters into STPBiomeRegistry
		 * @param biomeini The INI file for the biome config
		*/
		void loadBiomeParameters(const SuperTerrainPlus::STPAlgorithm::STPINIData::STPINIStorageView&);

	}
}
#endif//_STP_HEIGHTFIELD_PARA_LOADER_H_