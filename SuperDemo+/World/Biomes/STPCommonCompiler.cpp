#include "STPCommonCompiler.h"
#include <SuperAlgorithm+/STPAlgorithmDeviceInfo.h>

//Error
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//IO
#include <iostream>
#include <SuperTerrain+/Utility/STPFile.h>

#include <glm/gtc/type_ptr.hpp>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;
using namespace STPDemo;

using std::string;

using std::cout;
using std::cerr;
using std::endl;

using glm::vec2;
using glm::uvec2;
using glm::value_ptr;

//Include dir of the engines
const static string device_include = "-I " + string(SuperTerrainPlus::SuperAlgorithmPlus_DeviceInclude),
core_include = "-I " + string(SuperTerrainPlus::SuperTerrainPlus_CoreInclude);

STPCommonCompiler::STPCommonCompiler(const SuperTerrainPlus::STPEnvironment::STPChunkSetting& chunk, 
	const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) : 
	Dimension(chunk.MapSize), RenderingRange(chunk.RenderedChunk), SimplexPermutation(simplex_setting) {
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

	this->setupCommonGenerator();
	//compile individual source codes
	this->setupBiomefieldGenerator();
	this->setupSplatmapGenerator();

	//link all source codes
	this->finalise();
}

#define HANDLE_COMPILE(FUNC) \
try { \
	const string log = FUNC; \
	if (!log.empty()) { \
		cout << log << endl; \
	} \
} \
catch (const SuperTerrainPlus::STPException::STPCompilationError& error) { \
	cerr << error.what() << endl; \
	std::terminate(); \
}

void STPCommonCompiler::setupCommonGenerator() {
	//Filename
	constexpr static char CommonGeneratorFilename[] = "./Script/STPCommonGenerator.cu";

	//load source code
	const STPFile commongen_source(CommonGeneratorFilename);
	//set compiler options
	STPRuntimeCompilable::STPSourceInformation commongen_info = this->getCompilerOptions();
	//this common generator only contains a few shared variables
	commongen_info.NameExpression
		["STPCommonGenerator::Dimension"]
	["STPCommonGenerator::HalfDimension"]
	["STPCommonGenerator::RenderedDimension"]
	["STPCommonGenerator::Permutation"];
	//compile
	HANDLE_COMPILE(this->compileSource("STPCommonGenerator", *commongen_source, commongen_info))
	
}

void STPCommonCompiler::setupBiomefieldGenerator() {
	//File name of the generator script
	constexpr static char GeneratorFilename[] = "./Script/STPMultiHeightGenerator.cu";
	constexpr static char BiomePropertyFilename[] = "./STPBiomeProperty.hpp";

	//read script
	const STPFile multiheightfield_source(GeneratorFilename);
	const STPFile biomeprop_hdr(BiomePropertyFilename);
	//attach biome property
	this->attachHeader("STPBiomeProperty", *biomeprop_hdr);
	//attach source code and load up default compiler options, it returns a copy
	STPRuntimeCompilable::STPSourceInformation multiheightfield_info = this->getCompilerOptions();
	//we only need to adjust options that are unique to different sources
	multiheightfield_info.NameExpression
		//global function
		["generateMultiBiomeHeightmap"]
	//constant
	["BiomeTable"];
	//options are all set
	multiheightfield_info.ExternalHeader
		["STPBiomeProperty"];
	HANDLE_COMPILE(this->compileSource("STPMultiHeightGenerator", *multiheightfield_source, multiheightfield_info))
}

void STPCommonCompiler::setupSplatmapGenerator() {
	//File name of the generator script
	constexpr static char GeneratorFilename[] = "./Script/STPSplatmapGenerator.cu";

	//read source code
	const STPFile splatmap_source(GeneratorFilename);
	//load default compiler settings
	STPCommonCompiler::STPSourceInformation source_info = this->getCompilerOptions();
	source_info.NameExpression
		["SplatDatabase"]
	//global function
	["generateTextureSplatmap"];
	//compile
	HANDLE_COMPILE(this->compileSource("STPSplatmapGenerator", *splatmap_source, source_info))
}

void STPCommonCompiler::finalise() {
	//linker log output
	STPCommonCompiler::STPCompilerLog log;
	STPRuntimeCompilable::STPLinkerInformation link_info = this->getLinkerOptions(log);
	try {
		this->linkProgram(link_info, CU_JIT_INPUT_PTX);
		cout << log.linker_info_log << endl;
		cout << log.module_info_log << endl;

		//setup some variables
		CUdeviceptr dimension, half_dimension, rendered_dimension, perm;
		size_t dimensionSize, half_dimensionSize, rendered_dimensionSize, permSize;
		//source information
		const auto& name = this->retrieveSourceLoweredName("STPCommonGenerator");
		CUmodule program = this->getGeneratorModule();
		STPcudaCheckErr(cuModuleGetGlobal(&dimension, &dimensionSize, program, name.at("STPCommonGenerator::Dimension")));
		STPcudaCheckErr(cuModuleGetGlobal(&half_dimension, &half_dimensionSize, program, name.at("STPCommonGenerator::HalfDimension")));
		STPcudaCheckErr(cuModuleGetGlobal(&rendered_dimension, &rendered_dimensionSize, program, name.at("STPCommonGenerator::RenderedDimension")));
		STPcudaCheckErr(cuModuleGetGlobal(&perm, &permSize, program, name.at("STPCommonGenerator::Permutation")));
		//send data
		const vec2 halfDim = static_cast<vec2>(this->Dimension) / 2.0f;
		const uvec2 RenderedDim = this->RenderingRange * this->Dimension;
		STPcudaCheckErr(cuMemcpyHtoD(dimension, value_ptr(this->Dimension), dimensionSize));
		STPcudaCheckErr(cuMemcpyHtoD(half_dimension, value_ptr(halfDim), half_dimensionSize));
		STPcudaCheckErr(cuMemcpyHtoD(rendered_dimension, value_ptr(RenderedDim), rendered_dimensionSize));
		//note that we are copying permutation to device, the underlying pointers are managed by this class
		STPcudaCheckErr(cuMemcpyHtoD(perm, &(*this->SimplexPermutation), permSize));
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