#include "STPCommonCompiler.h"

#ifdef _DEBUG
#include <STPAlgorithmDeviceInfoDebug.h>
#else
//We ignored other two release modes
#include <SuperAlgorithm+DeviceInfoRelease.h>
#endif//_DEBUG

//System
#include <string>

using namespace SuperTerrainPlus::STPCompute;
using namespace STPDemo;

using std::string;

//Include dir of the engines
const static string device_include = "-I " + string(SuperTerrainPlus::SuperAlgorithmPlus_DeviceInclude),
core_include = "-I " + string(SuperTerrainPlus::SuperTerrainPlus_CoreInclude);

STPCommonCompiler::STPCommonCompiler() {
	//setup compiler options
	this->SourceInfo.Option
		["-std=c++17"]
		["-fmad=false"]
		["-arch=compute_75"]
		["-rdc=true"]
#ifdef _DEBUG
		["-G"]
		["-lineinfo"]
#endif
		["-maxrregcount=80"]
		//set include paths
		[core_include.c_str()]
		[device_include.c_str()];

	//setup linker options
	this->LinkInfo.LinkerOption
#ifdef _DEBUG
		//if debug has turned on, optimisation must be 0 or linker will be crying...
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)0u)
		(CU_JIT_GENERATE_DEBUG_INFO, (void*)1)
		(CU_JIT_GENERATE_LINE_INFO, (void*)1)
#else
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)3u)
#endif
		(CU_JIT_LOG_VERBOSE, (void*)1);
	
	//setup module options
#ifdef _DEBUG
	this->LinkInfo.ModuleOption
		(CU_JIT_GENERATE_DEBUG_INFO, (void*)1);
#endif

	//attach device algorithm library
	this->attachArchive("SuperAlgorithm+Device", SuperTerrainPlus::SuperAlgorithmPlus_DeviceLibrary);
}

STPRuntimeCompilable::STPSourceInformation STPCommonCompiler::getCompilerOptions() const {
	//return a copy of that, because one compiler may need to compile multiple sources
	return this->SourceInfo;
}

STPRuntimeCompilable::STPLinkerInformation STPCommonCompiler::getLinkerOptions(STPCompilerLog& log) const {
	//make a copy of the link info so we don't modify the original one
	STPRuntimeCompilable::STPLinkerInformation link_info = this->LinkInfo;
	//fill in logs
	link_info.LinkerOption
		(CU_JIT_INFO_LOG_BUFFER, log.linker_info_log)
		(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize)
		(CU_JIT_ERROR_LOG_BUFFER, log.linker_error_log)
		(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize);
	link_info.ModuleOption
		(CU_JIT_INFO_LOG_BUFFER, log.module_info_log)
		(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize)
		(CU_JIT_ERROR_LOG_BUFFER, log.module_error_log)
		(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize);

	return link_info;
}