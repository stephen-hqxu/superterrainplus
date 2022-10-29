#pragma once
#ifndef _STP_DEVICE_RUNTIME_BINARY_H_
#define _STP_DEVICE_RUNTIME_BINARY_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../Utility/STPNullablePrimitive.h"

//Container
#include <unordered_map>
#include <vector>
#include <string>

//CUDA
#include <nvrtc.h>

namespace SuperTerrainPlus {

	/**
	 * @brief STPDeviceRuntimeBinary provides a device-side runtime compilation toolkit powered by NVRTC.
	*/
	namespace STPDeviceRuntimeBinary {

		/**
		 * @brief Parameter sets for source compilation.
		*/
		struct STP_API STPSourceInformation {
		public:

			/**
			 * @brief A helper argument setter for easy configuration.
			*/
			struct STP_API STPSourceArgument {
			public:

				//Store multiple string arguments that can be recognised by CUDA functions.
				std::vector<std::string> StringArgument;

				/**
				 * @brief Add one argument.
				 * @param arg The adding string argument.
				 * @return The current argument object for easy chained function call.
				*/
				STPSourceArgument& operator[](const std::string&);

			};

			//The compiler flag to compile this program.
			//Provide empty vector means no option should be included.
			STPSourceArgument Option;
			//Due to the auto-mangling nature of CUDA, any global scope __device__ variable and __global__ function
			//	name will be mangled after complication.If one wishes to obtain the address of such pointers, mangled
			//	named must be used.
			//By providing name expressions, generator can guarantee to retrieve the mangled name after complication
			//	provided name expression has been provided here.
			//Provide empty vector if no name expression is needed.
			STPSourceArgument NameExpression;
			//Include name of any included external header in this source code.
			//Content of the header will be imported from the database, provided it has been attached by attachHeader().
			//Provide empty vector indicates no external header should be added to complication.
			STPSourceArgument ExternalHeader;

		};

		//Contains external headers to be added for the compilation.
		//Key: external header name, which will be used as the name during #include.
		//Value: the complete source of the header.
		typedef std::unordered_map<std::string, std::string> STPExternalHeaderSource;
		//Contains lowered name for a source program. Note that the mangled name is owned by the NVRTC program.
		//Key: original name, Value: mangled name.
		typedef std::unordered_map<std::string, std::string> STPLoweredName;

		/**
		 * @brief STPProgramDeleter deletes a NVRTC program instance.
		*/
		struct STP_API STPProgramDeleter {
		public:

			void operator()(nvrtcProgram) const;

		};
		//A smartly managed NVRTC compiled program
		using STPSmartProgram = STPUniqueResource<nvrtcProgram, nullptr, STPProgramDeleter>;
		//Raw program data retrieved from compiled program, and the length
		using STPProgramData = std::pair<std::unique_ptr<char[]>, size_t>;

		/**
		 * @brief STPCompilationOutput holds the output information of the compiled program.
		*/
		struct STPCompilationOutput {
		public:

			//The compiled NVRTC program object, can be referred as an object file.
			struct STPCompiledBinary {
			public:

				//A user-specified name for the program for debugging, can be an empty string.
				std::string Identifier;
				//The compiled program.
				STPSmartProgram Program;

			} ProgramObject;

			//The log from the compiler.
			std::string Log;
			//The lowered names of all registered variables and functions.
			//If name expression is added, valid name will be kept here.
			//Name which cannot be found in the source code will be discarded.
			STPLoweredName LoweredName;

		};

		/**
		 * @brief Given a piece of source code, compile it and create the NVRTC program.
		 * Any existing NVRTC program will be destroyed upon successful compilation.
		 * If compilation is unsuccessful, exception will be generated, and no change is made to the current program.
		 * @param source_name The name of the source file.
		 * The source name is used for debug purposes.
		 * @param source_code The actual code of the source.
		 * @param source_info The information for the compiler.
		 * @param external_header The addition information regarding any externally used headers.
		 * It is an error if header name is specified in the source information but header definition is not found here.
		 * Provide nullptr to indicate no use of external header.
		 * @return The compilation output.
		 * @see STPCompilationOutput
		*/
		STP_API STPCompilationOutput compile(std::string&&, const std::string&, const STPSourceInformation&,
			const STPExternalHeaderSource& = STPExternalHeaderSource());

		/**
		 * @brief Read PTX code from the underlying compiled program.
		 * @param program The program.
		 * @return PTX code associated with the given program.
		*/
		STP_API STPProgramData readPTX(nvrtcProgram);

		/**
		 * @brief Read the CUBIN binary from the underlying compiled program.
		 * @param program The program.
		 * @return CUBIN binary associated with the given program.
		*/
		STP_API STPProgramData readCUBIN(nvrtcProgram);
	}
}
#endif//_STP_DEVICE_RUNTIME_BINARY_H_