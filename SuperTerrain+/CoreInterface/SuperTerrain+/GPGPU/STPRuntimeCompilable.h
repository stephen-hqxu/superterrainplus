#pragma once
#ifndef _STP_RUNTIME_COMPILABLE_H_
#define _STP_RUNTIME_COMPILABLE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <memory>
//CUDA Runtime Compiler
#include <cuda.h>
#include <nvrtc.h>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPRuntimeCompilable provides a device-side runtime-programmable interface and toolsets with NVRTC.
	*/
	class STP_API STPRuntimeCompilable {
	protected:

		//Store multiple string arguments that can be recognised by CUDA functions
		typedef std::vector<const char*> STPStringArgument;
		//CUDA JIT flag for driver module
		typedef std::vector<CUjit_option> STPJitFlag;
		//CUDA JIT flag value
		typedef std::vector<void*> STPJitFlagValue;
		//Contains lowered name for a source program
		//Key: original name, Value: lowered name
		typedef std::unordered_map<std::string, const char*> STPLoweredName;
		//Contains lowered name for all registered source program
		//Key: source name, Value: STPLoweredName
		//@see STPLoweredName
		typedef std::unordered_map<std::string, STPLoweredName> STPNameExpression;

		/**
		 * @brief Parameter sets for source complication
		*/
		struct STP_API STPSourceInformation {
		public:

			friend class STPRuntimeCompilable;

			/**
			 * @brief A helper argument setter for easy configuration
			*/
			struct STP_API STPSourceArgument : private STPStringArgument {
			private:

				friend class STPRuntimeCompilable;

			public:

				/**
				 * @brief Add one argument
				 * @param arg The added argument
				 * @return The current argument object for easy chained function call
				*/
				STPSourceArgument& operator[](const char[]);

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
		struct STP_API STPLinkerInformation {
		public:

			friend class STPRuntimeCompilable;

			/**
			 * @brief Parameter sets for individual data in the linker
			*/
			struct STP_API STPDataJitOption {
			public:

				friend class STPRuntimeCompilable;

			private:

				//Individual override flag for some optional source files
				STPJitFlag OptionFlag;
				//Individual override respective flag value for some optional source files
				STPJitFlagValue OptionValue;

			public:

				/**
				 * @brief Emplace a new flag with value set for one data source
				 * @param flag Flag for this data
				 * @param value Value for this flag
				 * @return The current object for easy chained function call
				*/
				STPDataJitOption& operator()(CUjit_option, void*);

			};

		private:

			//Given individual option for optional source files.
			//It will override the global option set for that file.
			std::unordered_map<std::string, STPDataJitOption> DataOption;

		public:


			//The assembler and linker flag and value
			STPDataJitOption LinkerOption;

			/**
			 * @brief Get the data option for one source file
			 * @param source_name The name of the source file to get.
			 * If this source has yet had any options added, a new entry will be inserted.
			 * @return The data option for this source file
			*/
			STPDataJitOption& getDataOption(std::string);

			//The generator program module flag and value
			STPDataJitOption ModuleOption;
		};

	private:

		/**
		 * @brief Delete nvrtcProgram
		 * @param program The program to be deleted
		*/
		static void deleteProgram(nvrtcProgram);

		/**
		 * @brief Delete CUmodule
		 * @param module The module to be deleted
		*/
		static void deleteModule(CUmodule);

		/**
		 * @brief Delete CUlinkState
		 * @param link The link to be deleted
		*/
		static void deleteLink(CUlinkState);

		typedef std::unique_ptr<std::remove_pointer_t<nvrtcProgram>, void(*)(nvrtcProgram)> ManagednvrtcProgram;
		typedef std::unique_ptr<std::remove_pointer_t<CUmodule>, void(*)(CUmodule)> ManagedCUmodule;
		typedef std::unique_ptr<std::remove_pointer_t<CUlinkState>, void(*)(CUlinkState)> ManagedCUlinkState;

		//Store included files
		//Key: header name, Value: header code
		typedef std::unordered_map<std::string, std::string> STPIncluded;
		//Store compiled device source code in device format
		//Key: source name, Value: compiled code
		typedef std::unordered_map<std::string, ManagednvrtcProgram> STPCompiled;

		//All external header used within the runtime script that cannot be found directly under the script's directory nor the include directory set during complication
		//This can be useful for in-place code, i.e., code is a hard-coded string in the executable
		//Key: header include name, Value: header code
		STPIncluded ExternalHeader;
		//All precompiled static library that will be linked with the main generator program
		//Key: archive name, Value: the filename of the archive
		STPIncluded ExternalArchive;
		//All source files compiled in PTX format.
		STPCompiled CompilationDatabase;
		//All registered lowered name expression with programs.
		STPNameExpression CompilationNameDatabase;
		//A complete program of diversity generator
		ManagedCUmodule GeneratorProgram;

	protected:

		/**
		 * @brief Init a new STPDiversityGenerator
		*/
		STPRuntimeCompilable();

		STPRuntimeCompilable(const STPRuntimeCompilable&) = delete;

		STPRuntimeCompilable(STPRuntimeCompilable&&) = delete;

		STPRuntimeCompilable& operator=(const STPRuntimeCompilable&) = delete;

		STPRuntimeCompilable& operator=(STPRuntimeCompilable&&) = delete;

		/**
		 * @brief Read source code given a filename as input
		 * @param filename The file to be read.
		 * If filename cannot be found, exception is thrown
		 * @return The source code contained in file
		*/
		std::string readSource(std::string);

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
		 * @brief Attach an external archive into the linker database.
		 * Archive will be linked with the program
		 * @param archive_name The archive name used to identify this archive
		 * @param archive_filename The path to the archive
		 * @return Only return false if archive name is duplicate
		*/
		bool attachArchive(std::string, std::string);

		/**
		 * @brief Detach an external archive that has been previously attached to this linker database
		 * @param archive_name The name of the archive
		 * @return Only return false if archive is not found
		*/
		bool detachArchive(std::string);

		/**
		 * @brief Given a piece of source code, compile it and attach the compiled result to generator's database.
		 * If source with the same name has been compiled before, an exception will be thrown, and no operation is done.
		 * If one wishes to replace source with the same name, call discardSource() first.
		 * If name expression is added, only valid name will be kept in the name expression database, name which cannot be found in the source code will be discarded.
		 * @param source_name The name of the source file
		 * @param source_code The actual code of the source
		 * @param source_info The information for the compiler.
		 * @return log The log output from the compiler.
		 * If exception is thrown before complication, no log will be generated
		*/
		std::string compileSource(std::string, const std::string&, const STPSourceInformation&);

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
		 * If linking error occurs, the existing module will not be modified
		 * @param linker_info The information for the linker
		 * @param input_type The input type to be used. Currently only CU_JIT_INPUT_PTX, CU_JIT_INPUT_CUBIN is supported
		 * By using a low-level type, compile speed is faster.
		 * By using a high-level type, more compiler and linker options are available.
		*/
		void linkProgram(STPLinkerInformation&, CUjitInputType);

		/**
		 * @brief Retrieve mangled name for each name that has been added to name expression when the source was compiled.
		 * Only name expression that has been added prior to complication can be retrieved, otherwise exception is thrown
		 * @param source_name The name of the source that was used compiled and added to the database.
		 * If source_name cannot be found, exception is thrown.
		 * @return Given each key as the original name, value will be overwriten as the lowered name.
		 * Retrieved lowered name pointer is valid as long as the generator is not destroied and source is not discarded.
		*/
		const STPLoweredName& retrieveSourceLoweredName(std::string) const;

		/**
		 * @brief Get the complete generator program module.
		 * @return The generator module.
		 * If the program is not yet linked, or module has been unloaded, the returned result is undefined.
		*/
		CUmodule getGeneratorModule() const;

	public:

		virtual ~STPRuntimeCompilable() = default;

	};
}
#endif//_STP_RUNTIME_COMPILABLE_H_