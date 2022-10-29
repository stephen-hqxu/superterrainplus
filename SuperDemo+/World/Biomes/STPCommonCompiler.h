#pragma once
#ifndef _STP_COMMON_COMPILER_H_
#define _STP_COMMON_COMPILER_H_

//Runtime Compiler
#include <SuperTerrain+/GPGPU/STPDeviceRuntimeProgram.h>

//Setting
#include <SuperTerrain+/Environment/STPChunkSetting.h>
//Generator
#include <SuperAlgorithm+/STPPermutationGenerator.h>

//System
#include <string>

//GLM
#include <glm/vec2.hpp>

namespace STPDemo {

	/**
	 * @brief STPCommonCompiler is a even-higher-level wrapper to runtime compilable framework which provides default compiler settings that can be 
	 * shared across different translation units.
	*/
	class STPCommonCompiler {
	private:

		//The complete program after linking
		SuperTerrainPlus::STPDeviceRuntimeProgram::STPSmartModule GeneratorProgram;
		//Lowered name for each object file
		SuperTerrainPlus::STPDeviceRuntimeBinary::STPLoweredName BiomefieldName, SplatmapName;

		//all parameters for the noise generator, stored on host, passing value to device
		SuperTerrainPlus::STPAlgorithm::STPPermutationGenerator SimplexPermutation;

		//The chunk setting of each map used by each generator.
		const glm::uvec2 Dimension, RenderingRange;

	public:

		/**
		 * @brief Init STPCommonCompiler to its default state.
		 * SuperAlgorithm+Device library will be linked automatically.
		 * @param chunk The pointer to the chunk settings.
		 * @param simplex_setting The pointer to simplex noise setting.
		*/
		STPCommonCompiler(const SuperTerrainPlus::STPEnvironment::STPChunkSetting&, const SuperTerrainPlus::STPEnvironment::STPSimplexNoiseSetting&);

		STPCommonCompiler(const STPCommonCompiler&) = delete;

		STPCommonCompiler(STPCommonCompiler&&) = delete;

		STPCommonCompiler& operator=(const STPCommonCompiler&) = delete;

		STPCommonCompiler& operator=(STPCommonCompiler&&) = delete;

		~STPCommonCompiler() = default;

		/**
		 * @brief Get the linked program
		 * @return The module to the program
		*/
		CUmodule getProgram() const;
		
		//Get the lowered names for each program.
		const SuperTerrainPlus::STPDeviceRuntimeBinary::STPLoweredName& getBiomefieldName() const;
		const SuperTerrainPlus::STPDeviceRuntimeBinary::STPLoweredName& getSplatmapName() const;

	};

}
#endif//_STP_COMMON_COMPILER_H_