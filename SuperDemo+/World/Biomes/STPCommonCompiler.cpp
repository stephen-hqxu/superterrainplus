#include "STPCommonCompiler.h"

#ifdef _DEBUG
#include <STPAlgorithmDeviceInfoDebug.h>
#else
//We ignored other two release modes
#include <SuperAlgorithm+DeviceInfoRelease.h>
#endif//_DEBUG

//Error
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>

//IO
#include <iostream>

using namespace SuperTerrainPlus::STPCompute;
using namespace STPDemo;

using std::string;

using std::cout;
using std::cerr;
using std::endl;

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

	//compile individual source codes
	this->setupBiomefieldGenerator();
	//TODO: enable splatmap generator
	//this->setupSplatmapGenerator();

	//link all source codes
	this->finalise();
}

void STPCommonCompiler::setupBiomefieldGenerator() {
	//File name of the generator script
	constexpr static char GeneratorFilename[] = "./Script/STPMultiHeightGenerator.cu";
	constexpr static char BiomePropertyFilename[] = "./STPBiomeProperty.hpp";

	//read script
	const string multiheightfield_source = this->readSource(GeneratorFilename);
	const string biomeprop_hdr = this->readSource(BiomePropertyFilename);
	//attach biome property
	this->attachHeader("STPBiomeProperty", biomeprop_hdr);
	//attach source code and load up default compiler options, it returns a copy
	STPRuntimeCompilable::STPSourceInformation multiheightfield_info = this->getCompilerOptions();
	//we only need to adjust options that are unique to different sources
	multiheightfield_info.NameExpression
		//global function
		["generateMultiBiomeHeightmap"]
	//constant
	["BiomeTable"]
	["Permutation"]
	["Dimension"]
	["HalfDimension"];
	//options are all set
	multiheightfield_info.ExternalHeader
		["STPBiomeProperty"];
	try {
		const string log = this->compileSource("STPMultiHeightGenerator", multiheightfield_source, multiheightfield_info);
		if (!log.empty()) {
			cout << log << endl;
		}
	}
	catch (const SuperTerrainPlus::STPException::STPCUDAError& error) {
		cerr << error.what() << endl;
		std::terminate();
	}
}

void STPCommonCompiler::setupSplatmapGenerator() {
	//File name of the generator script
	constexpr static char GeneratorFilename[] = "./Script/STPSplatmapGenerator.cu";

	//read source code
	const string splatmap_source = this->readSource(GeneratorFilename);
	//load default compiler settings
	STPCommonCompiler::STPSourceInformation source_info = this->getCompilerOptions();
	source_info.NameExpression
		["SplatDatabase"]
	["MapDimension"]
	["TotalBufferDimension"]
	//global function
	["generateTextureSplatmap"];
	//compile
	try {
		const string log = this->compileSource("STPSplatmapGenerator", splatmap_source, source_info);
		if (!log.empty()) {
			cout << log << endl;
		}
	}
	catch (const SuperTerrainPlus::STPException::STPCUDAError& error) {
		cerr << error.what() << endl;
		std::terminate();
	}
}

void STPCommonCompiler::finalise() {
	//linker log output
	STPCommonCompiler::STPCompilerLog log;
	STPRuntimeCompilable::STPLinkerInformation link_info = this->getLinkerOptions(log);
	try {
		this->linkProgram(link_info, CU_JIT_INPUT_PTX);
		cout << log.linker_info_log << endl;
		cout << log.module_info_log << endl;
	}
	catch (const SuperTerrainPlus::STPException::STPCUDAError& error) {
		cerr << error.what() << std::endl;
		cerr << log.linker_error_log << endl;
		cerr << log.module_error_log << endl;
		std::terminate();
	}
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

CUmodule STPCommonCompiler::getProgram() const {
	return this->getGeneratorModule();
}

const STPRuntimeCompilable::STPLoweredName& STPCommonCompiler::getLoweredNameDictionary(const string& name) const {
	return this->retrieveSourceLoweredName(name);
}