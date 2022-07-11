#pragma once
#ifndef _STP_DEVICE_RUNTIME_PROGRAM_H_
#define _STP_DEVICE_RUNTIME_PROGRAM_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "STPDeviceRuntimeBinary.h"

//Container
#include <vector>
#include <unordered_map>
#include <tuple>

//CUDA
#include <cuda.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPDeviceRuntimeProgram provides a runtime-linking interface for compiled binary code, powered by CUDA.
	*/
	class STP_API STPDeviceRuntimeProgram {
	public:

		/**
		 * @brief STPBinaryType specifies the type of binary after compilation.
		*/
		enum class STPBinaryType : unsigned char {
			//The compiler should output PTX code.
			PTX = 0x00u,
			//The compiler should output CUBIN binary.
			CUBIN = 0xFFu
		};

		/**
		 * @brief Parameter sets for linker.
		*/
		struct STP_API STPLinkerInformation {
		public:

			/**
			 * @brief Parameter sets for individual data in the linker.
			*/
			struct STP_API STPDataJitOption {
			public:

				//Make sure they have the same size, and each flag corresponds to one value.
				//Individual override flag for some optional source files.
				std::vector<CUjit_option> OptionFlag;
				//Individual override respective flag value for some optional source files.
				std::vector<void*> OptionValue;

				/**
				 * @brief Emplace a new flag with value set for one data source.
				 * @param flag Flag for this data.
				 * @param value Value for this flag.
				 * @return The current object for easy chained function call.
				*/
				STPDataJitOption& operator()(CUjit_option, void*);

			};

			//The assembler and linker flag and value.
			STPDataJitOption LinkerOption;
			//Given individual option for optional source files.
			//It will override the global option set for that file.
			//Also specifies the input type of the binary.
			std::vector<std::tuple<const STPDeviceRuntimeBinary*, STPBinaryType, STPDataJitOption>> DataOption;
			//All pre-compiled static library that will be linked with the main generator program.
			//Should provide the filename to locate the archive, and the individual option.
			std::vector<std::pair<std::string, STPDataJitOption>> ArchiveOption;
			//The generator program module flag and value.
			STPDataJitOption ModuleOption;

		};

	private:

		/**
		 * @brief STPModuleDeleter deletes a CUDA module.
		*/
		struct STPModuleDeleter {
		public:

			void operator()(CUmodule) const;

		};
		typedef std::unique_ptr<std::remove_pointer_t<CUmodule>, STPModuleDeleter> STPManagedModule;
		STPManagedModule Module;

	public:

		STPDeviceRuntimeProgram() = default;

		STPDeviceRuntimeProgram(const STPDeviceRuntimeProgram&) = delete;

		STPDeviceRuntimeProgram(STPDeviceRuntimeProgram&&) = default;

		STPDeviceRuntimeProgram& operator=(const STPDeviceRuntimeProgram&) = delete;

		STPDeviceRuntimeProgram& operator=(STPDeviceRuntimeProgram&&) = default;

		~STPDeviceRuntimeProgram() = default;

		/**
		 * @brief Get the linked module.
		 * @return The linked module.
		 * If the program is incomplete, nullptr is returned.
		*/
		CUmodule operator*() const;

		/**
		 * @brief Link all provided binaries into a complete program.
		 * If there has been a program currently associated , it will be destroyed and the new one will be loaded,
		 * only if linking is successful.
		 * If linking error occurs, the existing module will not be modified.
		 * @param linker_info The information for the linker.
		*/
		void linkFromBinary(STPLinkerInformation&);

	};
}
#endif//_STP_DEVICE_RUNTIME_PROGRAM_H_