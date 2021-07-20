#pragma once
#ifndef _STP_DIVERSITY_GENERATOR_H_
#define _STP_DIVERSITY_GENERATOR_H_

//System
#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <memory>
//CUDA Runtime Compiler
#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>
//Biome Defines
#include "../World/Biome/STPBiomeDefine.h"

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
		 * @brief STPDiversityGenerator provides a programmable multi-biome heightmap generation interface and
		 * allows users to develop their biome-specific algorithms and parameters sets.
		*/
		class STPDiversityGenerator {
		private:

			//Store included files
			//Key: header name, Value: header code
			typedef std::unordered_map<std::string, std::string> STPIncluded;
			//Store compiled device source code in device format
			//Key: source name, Value: compiled code
			typedef std::unordered_map<std::string, nvrtcProgram> STPCompiled;

			//All external header used within the runtime script that cannot be found directly under the script's directory nor the include directory set during complication
			//This can be useful for in-place code, i.e., code is a hard-coded string in the executable
			STPIncluded ExternalHeader;
			//All source files compiled in PTX format.
			STPCompiled ComplicationDatabase;
			//A complete program of diversity generator
			CUmodule GeneratorProgram;
			bool ModuleLoadingStatus;
			//Indicate the global function address we need to call when generation is requested
			CUfunction Entry;

		protected:

			//Store multiple string arguments that can be recognised by CUDA functions
			typedef std::vector<const char*> STPStringArgument;
			//CUDA JIT flag for driver module
			typedef std::vector<CUjit_option> STPJitFlag;
			//CUDA JIT flag value
			typedef std::vector<void*> STPJitFlagValue;
			//Individual override flag for some optional source files
			//Key: source filename, Value: pair of argument and respective value
			typedef std::unordered_map<std::string, std::pair<STPJitFlag, STPJitFlagValue>> STPDataArgument;

			/**
			 * @brief Init a new STPDiversityGenerator
			*/
			STPDiversityGenerator();

			STPDiversityGenerator(const STPDiversityGenerator&) = delete;

			STPDiversityGenerator(STPDiversityGenerator&&) = delete;

			STPDiversityGenerator& operator=(const STPDiversityGenerator&) = delete;

			STPDiversityGenerator& operator=(STPDiversityGenerator&&) = delete;

			/**
			 * @brief Attach an external header into this runtime compiler database.
			 * Only headers that are not able to be found in the include path should be added,
			 * or any header rename, i.e., include name is not the same as filename.
			 * @param header_name The name by which the header is #included
			 * @param header_code The actual source code of the header
			 * @return True if header is attached.
			 * False if a duplicate header name is found.
			*/
			bool attachHeader(std::string, const std::string&);

			/**
			 * @brief Detach an external header that has been previously attached to this generator database
			 * @param header_name The name of the header to be detached.
			 * @return Only return false if header is not found.
			*/
			bool detachHeader(std::string);

			/**
			 * @brief Given a piece of source code, compile it and attach the compiled result to generator's database.
			 * @param source_name The name of the source file
			 * @param source_code The actual code of the source
			 * @param option The compiler flag to compile this program.
			 * Provide empty vector means no option should be included
			 * @param name_expression Due to the auto-mangling nature of CUDA, any global scope __device__ variable and __global__ function name
			 * will be mangled after complication. If one wishes to obtain the address of such pointers, mangled named must be used.
			 * By providing name expressions, generator can guarantee to retrieve the mangled name after complication provided name expression 
			 * has been provided here.
			 * Providew empty vector if no name expression is needed
			 * @param extern_header Include name of any included external header in this source code.
			 * Content of the header will be imported from the database, provided it has been attached by attachHeader().
			 * Provide empty vector indicates no external header should be added to complication.
			 * @return The log of the compiler
			*/
			std::unique_ptr<char[]> compileSource(std::string, const std::string&, const STPStringArgument&, const STPStringArgument&, const STPStringArgument&);

			/**
			 * @brief Discard previously compiled source file.
			 * @param source_name The name of the source file.
			 * @return If name doesn't exist in generator database, false is returned.
			 * Otherwise it will always return true.
			*/
			bool discardSource(std::string);

			/**
			 * @brief Link all previously compiled source file into a complete program.
			 * @param option_flag The assembler and linker flag
			 * @param option_value The respective values for the flag
			 * @param data_option Given individual option for optional source files.
			 * It will override the global option set for that file.
			*/
			std::unique_ptr<char[]> linkProgram(const STPJitFlag&, const STPJitFlagValue&, const STPDataArgument&);

		public:

			virtual ~STPDiversityGenerator();

			/**
			 * @brief Generate a biome-specific heightmaps
			 * @param heightmap The result of generated heightmap that will be stored
			 * @param biomemap The biomemap, which is an array of biomeID, the meaning of biomeID is however implementation-specific
			 * @param offset The offset of maps in world coordinate
			*/
			virtual void operator()(float*, const STPDiversity::Sample*, float2) = 0;

		};
	}
}
#endif//_STP_DIVERSITY_GENERATOR_H_