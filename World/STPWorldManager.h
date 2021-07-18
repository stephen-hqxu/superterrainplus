#pragma once
#ifndef _STP_WORLD_MANAGER_H_
#define _STP_WORLD_MANAGER_H_

//World
#include "STPProcedural2DINF.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPWorldManager is a high-level binding that allows user to generate and render any infinite procedural world within minimal efforts.
	*/
	class STPWorldManager {
	private:

		//Denote the linking status of the current world manager.
		bool linkStatus;

		//All stuff below are trival, no documentation is needed
		//settings
		std::unique_ptr<STPSettings::STPConfigurations> WorldSettings;
		//generators
		std::unique_ptr<STPCompute::STPHeightfieldGenerator> ChunkGenerator;
		std::unique_ptr<STPDiversity::STPBiomeFactory> BiomeFactory;
		//world management agents
		std::unique_ptr<STPChunkStorage> ChunkStorage;
		std::unique_ptr<STPChunkProvider> ChunkProvider;
		std::unique_ptr<STPChunkManager> ChunkManager;
		std::unique_ptr<STPProcedural2DINF> WorldRenderer;

	public:

		/**
		 * @brief Init the world manager
		*/
		STPWorldManager();

		~STPWorldManager() = default;

		STPWorldManager(const STPWorldManager&) = delete;

		STPWorldManager(const STPWorldManager&&) = delete;

		const STPWorldManager operator=(const STPWorldManager&) = delete;

		const STPWorldManager operator=(const STPWorldManager&&) = delete;

		/**
		 * @brief Attach world settings to the current world manager.
		 * @param settings All world settings, it will be copied under the object
		*/
		void attachSettings(STPSettings::STPConfigurations*);

		/**
		 * @brief Attach the biome factory with this world manager
		 * @tparam ...Arg Argument to create a concrete instance of biome factory
		 * @param arg... Parameter set to create a concrete instance of biome factory
		*/
		template<class Fac, typename... Arg>
		void attachBiomeFactory(Arg&&...);

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
		const STPSettings::STPConfigurations* getWorldSettings() const;

		/**
		 * @brief Get the chunk generator.
		 * @return The heightfield generator managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPCompute::STPHeightfieldGenerator* getChunkGenerator() const;

		/**
		 * @brief Get the biome factory
		 * @return The biome factory managed by the current world manager. If no biome factory is attached, nullptr is returned
		*/
		const STPDiversity::STPBiomeFactory* getBiomeFactory() const;

		/**
		 * @brief Get the chunk storage
		 * @return The chunk storage managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPChunkStorage* getChunkStorage() const;

		/**
		 * @brief Get the chunk provider
		 * @return The chunk provider managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPChunkProvider* getChunkProvider() const;

		/**
		 * @brief Get the chunk manager
		 * @return The chunk manager managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPChunkManager* getChunkManager() const;

		/**
		 * @brief Get the chunk renderer
		 * @return The infinite terrain renderer managed by the current world manager. If world manager is not linked, nullptr is returned.
		*/
		const STPProcedural2DINF* getChunkRenderer() const;

	};

}
#include "STPWorldManager.inl"
#endif//_STP_WORLD_MANAGER_H_