#pragma once
#ifndef _STP_WORLD_MANAGER_H_
#define _STP_WORLD_MANAGER_H_

//World
#include "STPProcedural2DINF.h"
//Compiler
#include "./Biomes/STPCommonCompiler.h"

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
		//settings
		SuperTerrainPlus::STPEnvironment::STPConfiguration WorldSetting;
		//generators
		std::optional<SuperTerrainPlus::STPCompute::STPHeightfieldGenerator> ChunkGenerator;
		std::unique_ptr<SuperTerrainPlus::STPDiversity::STPBiomeFactory> BiomeFactory;
		std::unique_ptr<SuperTerrainPlus::STPCompute::STPDiversityGenerator> DiversityGenerator;
		//world management agents
		std::optional<SuperTerrainPlus::STPChunkStorage> ChunkStorage;
		//make sure provider is destroyed (it will auto sync) before all generators and storage because it's the multi-threaded commander to call all other generators
		std::optional<SuperTerrainPlus::STPChunkProvider> ChunkProvider;
		std::optional<SuperTerrainPlus::STPChunkManager> ChunkManager;
		std::optional<STPProcedural2DINF> WorldRenderer;

	public:

		/**
		 * @brief Init the world manager.
		 * @param tex_filename_prefix The prefix for all texture filenames. It will be used as "<prefix>/<texture filename>".
		 * @param settings All world settings, it will be moved under the object
		*/
		STPWorldManager(std::string, SuperTerrainPlus::STPEnvironment::STPConfiguration&);

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
		 * @brief Link all pipeline stages together
		 * @param indirect_cmd The indrect rendering command for renderer
		*/
		void linkProgram(void*);

		/**
		 * @brief Get the link status of the current world manager.
		 * All generators will not be available unless linkStatus is true.
		*/
		operator bool() const;

		/**
		 * @brief Get the world settings 
		 * @return The world settings managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const SuperTerrainPlus::STPEnvironment::STPConfiguration& getWorldSetting() const;

		/**
		 * @brief Get the chunk generator.
		 * @return The heightfield generator managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const SuperTerrainPlus::STPCompute::STPHeightfieldGenerator& getChunkGenerator() const;

		/**
		 * @brief Get the biome factory
		 * @return The biome factory managed by the current world manager. If no biome factory is attached, nullptr is returned
		*/
		const SuperTerrainPlus::STPDiversity::STPBiomeFactory& getBiomeFactory() const;

		/**
		 * @brief Get the chunk storage
		 * @return The chunk storage managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const SuperTerrainPlus::STPChunkStorage& getChunkStorage() const;

		/**
		 * @brief Get the chunk provider
		 * @return The chunk provider managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const SuperTerrainPlus::STPChunkProvider& getChunkProvider() const;

		/**
		 * @brief Get the chunk manager
		 * @return The chunk manager managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const SuperTerrainPlus::STPChunkManager& getChunkManager() const;

		/**
		 * @brief Get the chunk renderer
		 * @return The infinite terrain renderer managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPProcedural2DINF& getChunkRenderer() const;

	};

}
#include "STPWorldManager.inl"
#endif//_STP_WORLD_MANAGER_H_