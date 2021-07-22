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
#include <cuda_runtime.h>
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
		 * @brief STPDiversityGenerator provides a runtime-programmable multi-biome heightmap generation interface and
		 * allows users to develop their biome-specific algorithms and parameters sets.
		*/
		class STPDiversityGenerator {
		protected:

			//Log message from the generator compiler and linker
			typedef std::unique_ptr<char[]> STPGeneratorLog;
			//Store multiple string arguments that can be recognised by CUDA functions
			typedef std::vector<const char*> STPStringArgument;
			//CUDA JIT flag for driver module
			typedef std::vector<CUjit_option> STPJitFlag;
			//CUDA JIT flag value
			typedef std::vector<void*> STPJitFlagValue;
			//Contains lowered name for a source program
			//Key: original name, Value: lowered name
			typedef std::unordered_map<std::string, const char*> STPLoweredName;

			/**
			 * @brief Parameter sets for source complication
			*/
			struct STPSourceInformation {
			public:

				friend class STPDiversityGenerator;

				/**
				 * @brief A helper argument setter for easy configuration
				*/
				struct STPSourceArgument : private STPStringArgument {
				public:

					friend class STPDiversityGenerator;

				public:

					/**
					 * @brief Add one argument
					 * @param arg The added argument
					 * @return The current argument object for easy chained function call
					*/
					STPSourceArgument& addArg(const char[]);

				};

				//The compiler flag to compile this program.
				//Provide empty vector means no option should be included
				STPSourceArgument Option;
				//- Due to the auto-mangling nature of CUDA, any global scope __device__ variable and __global__ function name
				//will be mangled after complication.If one wishes to obtain the address of such pointers, mangled named must be used.
				//- By providing name expressions, generator can guarantee to retrieve the mangled name after complication provided name expression has been provided here.
				//- Providew empty vector if no name expression is needed
				STPSourceArgument NameExpression;
				//Include name of any included external header in this source code.
				//Content of the header will be imported from the database, provided it has been attached by attachHeader().
				//Provide empty vector indicates no external header should be added to complication.
				STPSourceArgument ExternalHeader;

			};

			/**
			 * @brief Parameter sets for linker
			*/
			struct STPLinkerInformation {
			public:

				friend class STPDiversityGenerator;

				/**
				 * @brief Parameter sets for individual data in the linker
				*/
				struct STPDataJitOption {
				public:

					friend class STPDiversityGenerator;

				private:

					//Individual override flag for some optional source files
					STPJitFlag DataOptionFlag;
					//Individual override respective flag value for some optional source files
					STPJitFlagValue DataOptionFlagValue;

				public:

					/**
					 * @brief Emplace a new flag with value set for one data source
					 * @param flag Flag for this data
					 * @param value Value for this flag
					 * @return The current object for easy chained function call
					*/
					STPDataJitOption& setDataOption(CUjit_option, void*);

				};

			private:

				//The assembler and linker flag
				STPJitFlag OptionFlag;
				//The respective values for the flag
				STPJitFlagValue OptionFlagValue;
				//Given individual option for optional source files.
				//It will override the global option set for that file.
				std::unordered_map<std::string, STPDataJitOption> DataOption;
				//The generator program module flag
				STPJitFlag ModuleOptionFlag;
				//The generator program module flag value
				STPJitFlagValue ModuleOptionFlagValue;

			public:

				/**
				 * @brief Emplace a new linker option and value
				 * @param flag The flag for the linker
				 * @param value The value for this flag
				 * @return The current object for easy chained function call
				*/
				STPLinkerInformation& setLinkerOption(CUjit_option, void*);

				/**
				 * @brief Emplace a new module loading option and value
				 * @param flag The flag for the module loader
				 * @param value The value for this flag
				 * @return The current object for easy chained function call
				*/
				STPLinkerInformation& setModuleLoadOption(CUjit_option, void*);

				/**
				 * @brief Get the data option for one source file
				 * @param source_name The name of the source file to get.
				 * If this source has yet had any options added, a new entry will be inserted.
				 * @return The data option for this source file
				*/
				STPDataJitOption& getDataOption(std::string);
			};

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

		protected:

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
			 * @param source_info The information for the compiler.
			 * @return The log of the compiler
			*/
			STPGeneratorLog compileSource(std::string, const std::string&, const STPSourceInformation&);

			/**
			 * @brief Discard previously compiled source file.
			 * @param source_name The name of the source file.
			 * @return If name doesn't exist in generator database, false is returned.
			 * Otherwise it will always return true.
			*/
			bool discardSource(std::string);

			/**
			 * @brief Link all previously compiled source file into a complete program.
			 * If there has been a program currently associated with this generator, it will be destroied and the new one will be loaded.
			 * @param linker_info The information for the linker
			*/
			void linkProgram(STPLinkerInformation&);

			/**
			 * @brief Retrieve mangled name for each name that has been added to name expression when the source was compiled.
			 * Only name expression that has been added prior to complication can be retrieved, otherwise exception is thrown
			 * @param source_name The name of the source that was used compiled and added to the database
			 * @param expression Given each key as the original name, value will be overwriten as the lowered name.
			 * Retrieved lowered name pointer is valid as long as the generator is not destroied and source is not discarded.
			 * @return If compiled source is not found, return false and nothing will be writen
			 * Otherwise true is returned.
			*/
			bool retrieveSourceLoweredName(std::string, STPLoweredName&) const;

			/**
			 * @brief Get the complete generator program module.
			 * @return The generator module.
			 * If the program is not yet linked, or module has been unloaded, the returned result is undefined.
			*/
			CUmodule getGeneratorModule() const;

		public:

			virtual ~STPDiversityGenerator();

			/**
			 * @brief Generate a biome-specific heightmaps
			 * @param heightmap The result of generated heightmap that will be stored
			 * @param biomemap The biomemap, which is an array of biomeID, the meaning of biomeID is however implementation-specific
			 * @param offset The offset of maps in world coordinate
			 * @param stream The stream currently being used
			*/
			virtual void operator()(float*, const STPDiversity::Sample*, float2, cudaStream_t) = 0;

		};
	}
}
#endif//_STP_DIVERSITY_GENERATOR_H_