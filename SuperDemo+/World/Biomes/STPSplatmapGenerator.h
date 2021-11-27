#pragma once
#ifndef _STP_SPLATMAP_GENERATOR_H_
#define _STP_SPLATMAP_GENERATOR_H_

//Base Splatmap Generator
#include <SuperTerrain+/World/Diversity/Texture/STPTextureFactory.h>
//Runtime Compiler
#include "STPCommonCompiler.h"

namespace STPDemo {

	/**
	 * @brief STPSplatmapGenerator provides implementations for generating rule-based biome-dependednt texture splatmap
	*/
	class STPSplatmapGenerator final : public SuperTerrainPlus::STPDiversity::STPTextureFactory {
	private:

		const STPCommonCompiler& KernelProgram;
		CUfunction SplatmapEntry;

		/**
		 * @brief Initialise runtimer compiler for splatmap generation
		*/
		void initGenerator();

		void splat(cudaTextureObject_t, cudaTextureObject_t, cudaSurfaceObject_t, 
			const SuperTerrainPlus::STPDiversity::STPTextureInformation::STPSplatGeneratorInformation&, cudaStream_t) const override;

	public:

		/**
		 * @brief Initialise splatmap generator.
		 * @param program The program contains the kernel source codes
		 * @param database_view A view to the texture database that has all texture data and splat rules loaded.
		 * @param chunk_setting The pointer to the chunk setting.
		*/
		STPSplatmapGenerator(const STPCommonCompiler& program, const SuperTerrainPlus::STPDiversity::STPTextureDatabase::STPDatabaseView&, 
			const SuperTerrainPlus::STPEnvironment::STPChunkSetting&);

		STPSplatmapGenerator(const STPSplatmapGenerator&) = delete;

		STPSplatmapGenerator(STPSplatmapGenerator&&) = delete;

		STPSplatmapGenerator& operator=(const STPSplatmapGenerator&) = delete;

		STPSplatmapGenerator& operator=(STPSplatmapGenerator&&) = delete;

		~STPSplatmapGenerator() = default;

	};

}
#endif//_STP_SPLATMAP_GENERATOR_H_