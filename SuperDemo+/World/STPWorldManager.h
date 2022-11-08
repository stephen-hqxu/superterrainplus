#pragma once
#ifndef _STP_WORLD_MANAGER_H_
#define _STP_WORLD_MANAGER_H_

//World
#include <SuperTerrain+/Environment/STPChunkSetting.h>
#include <SuperTerrain+/Environment/STPHeightfieldSetting.h>
//Compiler
#include "./Biomes/STPCommonCompiler.h"

//World Engine
#include <SuperTerrain+/World/STPWorldPipeline.h>
#include <SuperTerrain+/World/Chunk/STPHeightfieldGenerator.h>

//System
#include <optional>
#include <memory>

namespace STPDemo {

	/**
	 * @brief STPWorldManager is a high-level binding that allows user to generate and render any infinite procedural world within minimal efforts.
	*/
	class STPWorldManager {
	public:

		//A compiler contains all runtime scripts
		const STPDemo::STPCommonCompiler SharedProgram;

	private:

		//Denote the linking status of the current world manager.
		bool linkStatus;

		/**
		 * @brief STPWorldSplattingAgent manages external texture data so they can be used for terrain texturing,
		 * as well as building splat rules and database.
		*/
		class STPWorldSplattingAgent;
		std::unique_ptr<STPWorldSplattingAgent> Texture;

		//Order of declaration is very important
		//generators
		std::optional<SuperTerrainPlus::STPHeightfieldGenerator> ChunkGenerator;
		std::unique_ptr<SuperTerrainPlus::STPDiversity::STPBiomeFactory> BiomeFactory;
		std::unique_ptr<SuperTerrainPlus::STPDiversityGenerator> DiversityGenerator;
		std::unique_ptr<SuperTerrainPlus::STPDiversity::STPTextureFactory> TextureFactory;
		//make sure pipeline is destroyed (it will auto sync) before all generators and storage because it's the multi-threaded commander to call all other generators
		std::optional<SuperTerrainPlus::STPWorldPipeline> Pipeline;

		/**
		 * @brief Attach a type of custom attachment.
		 * @tparam Base The base instance where the attachment inherited from.
		 * @tparam Ins The instance of attachment to be added to the world manager.
		 * @tparam ...Arg Arguments to create a concrete instance of specific type of attachment.
		 * @param arg... Parameters to create a concrete instance of attachment.
		 * @return The smart pointer to the concrete instance in base class.
		*/
		template<class Base, class Ins, typename... Arg>
		auto attach(Arg&&...);

	public:

		/**
		 * @brief Init the world manager.
		 * @param tex_filename_prefix The prefix for all texture filenames. It will be used as "<prefix>/<texture filename>".
		 * @param chunk_setting All chunk settings.
		 * @param simplex_setting Setting for simplex noise generator.
		*/
		STPWorldManager(const std::string&, const SuperTerrainPlus::STPEnvironment::STPChunkSetting&,
			const SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting&);

		~STPWorldManager();

		STPWorldManager(const STPWorldManager&) = delete;

		STPWorldManager(STPWorldManager&&) = delete;

		STPWorldManager& operator=(const STPWorldManager&) = delete;

		STPWorldManager& operator=(STPWorldManager&&) = delete;

		/**
		 * @brief Attach the biome factory with this world manager
		 * @tpara Fac The instance of biome factory
		 * @tparam ...Arg Argument to create a concrete instance of biome factory
		 * @param arg... Parameter set to create a concrete instance of biome factory
		*/
		template<class Fac, typename... Arg>
		void attachBiomeFactory(Arg&&...);

		/**
		 * @brief Attach the multi-biome heightfield generator
		 * @tparam Div The instance of the generator
		 * @tparam ...Arg Argument to create a concrete instance of diversity generator
		 * @param arg... Parameters to create a concrete instance of diversity generator
		*/
		template<class Div, typename... Arg>
		void attachDiversityGenerator(Arg&&...);

		/**
		 * @brief Attach the texture factory with this world manager.
		 * @tparam Tex The instance of the texture factory.
		 * @tparam ...Arg Arguments to create a concrete instance of the texture factory.
		 * @param arg... Parameter set to create a concrete instance of texture factory.
		*/
		template<class Tex, typename... Arg>
		void attachTextureFactory(Arg&&...);

		/**
		 * @brief Link all pipeline stages together.
		 * @param anisotropy The level of anisotropy filtering to be used for textures.
		 * @param chunk_setting All chunk settings, it will be copied under the object.
		 * @param heightfield_setting All heightfield settings, also copied.
		*/
		void linkProgram(float, const SuperTerrainPlus::STPEnvironment::STPChunkSetting&,
			const SuperTerrainPlus::STPEnvironment::STPHeightfieldSetting&);

		/**
		 * @brief Get the link status of the current world manager.
		 * All generators will not be available unless linkStatus is true.
		*/
		operator bool() const;

		/**
		 * @brief Get a view to the texture database with preload settings.
		 * @return The view to the texture database.
		*/
		SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPDatabaseView getTextureDatabase() const;

		/**
		 * @brief Get the world pipeline
		 * @return The world pipeline managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		SuperTerrainPlus::STPWorldPipeline& getPipeline();

	};

}
#include "STPWorldManager.inl"
#endif//_STP_WORLD_MANAGER_H_