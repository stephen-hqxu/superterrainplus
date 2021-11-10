#pragma once
#ifndef _STP_COMMON_COMPILER_H_
#define _STP_COMMON_COMPILER_H_

//Runtime Compiler
#include <SuperTerrain+/GPGPU/STPRuntimeCompilable.h>

namespace STPDemo {

	/**
	 * @brief STPCommonCompiler is a even-higher-level wrapper to runtime compilable framework which provides default compiler settings that can be 
	 * shared across different translation units.
	*/
	class STPCommonCompiler : protected SuperTerrainPlus::STPCompute::STPRuntimeCompilable {
	private:

		//Contains compiler options (only) for source codes, external headers and named expr are not set
		STPRuntimeCompilable::STPSourceInformation SourceInfo;
		//Contains linker options
		STPRuntimeCompilable::STPLinkerInformation LinkInfo;

	protected:

		/**
		 * @brief STPCompilerLog contains allocated memory for compiler and linker logs
		*/
		struct STPCompilerLog {
		public:

			constexpr static unsigned int LogSize = 1024u;

			//Various of logs
			char linker_info_log[LogSize], linker_error_log[LogSize];
			char module_info_log[LogSize], module_error_log[LogSize];

		};

		/**
		 * @brief Init STPCommonCompiler to its default state.
		 * SuperAlgorithm+Device library will be linked automatically
		*/
		STPCommonCompiler();

		/**
		 * @brief Get common compiler options.
		 * Include directories are set to algorithm device library and core engine
		 * @return A copy of source information with default compiler options set, such that the returned source info does not affect the default settings.
		*/
		STPRuntimeCompilable::STPSourceInformation getCompilerOptions() const;

		/**
		 * @brief Get the common linker options.
		 * It also initialises linker message and error report.
		 * @param log The memory to allocated memory for returning linker logs.
		 * @return A copy of linker information with default options set
		*/
		STPRuntimeCompilable::STPLinkerInformation getLinkerOptions(STPCompilerLog&) const;

	public:

		STPCommonCompiler(const STPCommonCompiler&) = delete;

		STPCommonCompiler(STPCommonCompiler&&) = delete;

		STPCommonCompiler& operator=(const STPCommonCompiler&) = delete;

		STPCommonCompiler& operator=(STPCommonCompiler&&) = delete;

		~STPCommonCompiler() = default;

	};

}
#endif//_STP_COMMON_COMPILER_H_