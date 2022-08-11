#include "STPCommonCompiler.h"
#include <SuperTerrain+/STPCoreInfo.h>
#include <SuperAlgorithm+/STPAlgorithmDeviceInfo.h>

#include <SuperTerrain+/STPEngineInitialiser.h>

//Error
#include <SuperTerrain+/Exception/STPCompilationError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//IO
#include <iostream>
#include <SuperTerrain+/Utility/STPFile.h>

#include <array>

#include <glm/gtc/type_ptr.hpp>

using namespace SuperTerrainPlus;
using namespace STPDemo;

using std::to_string;
using std::string;
using std::string_view;
using std::array;

using std::cout;
using std::cerr;
using std::endl;

using glm::vec2;
using glm::uvec2;
using glm::value_ptr;

/**
 * @brief STPCompilerLog contains allocated memory for compiler and linker logs
*/
struct STPCompilerLog {
public:

	constexpr static size_t LogSize = 1024u;

	//Various of logs
	char linker_info_log[LogSize], linker_error_log[LogSize];
	char module_info_log[LogSize], module_error_log[LogSize];
};

#define HANDLE_COMPILE(FUNC) \
STPDeviceRuntimeBinary::STPCompilationOutput output; \
try { \
	output = FUNC; \
	if (!output.Log.empty()) { \
		cout << output.Log << endl; \
	} \
} catch (const SuperTerrainPlus::STPException::STPCompilationError& error) { \
	cerr << error.what() << endl; \
	std::terminate(); \
}

STPCommonCompiler::STPCommonCompiler(const SuperTerrainPlus::STPEnvironment::STPChunkSetting& chunk,
	const STPEnvironment::STPSimplexNoiseSetting& simplex_setting) : SimplexPermutation(simplex_setting),
	Dimension(chunk.MapSize), RenderingRange(chunk.RenderedChunk) {
	const auto commonSourceInfo = [capabilityOption = string("-arch=sm_") + to_string(STPEngineInitialiser::architecture(0))]() {
		STPDeviceRuntimeBinary::STPSourceInformation info;
		//setup compiler options
		info.Option
			["-std=c++17"]
			[capabilityOption]
			["-rdc=true"]
#ifndef NDEBUG
			["-G"]
#endif
			["-maxrregcount=80"]
			//set include paths
			["-I " + string(STPCoreInfo::CoreInclude)]
			["-I " + string(STPAlgorithm::STPAlgorithmDeviceInfo::DeviceInclude)];

		return info;
	}();

	STPDeviceRuntimeBinary bin_common, bin_biome, bin_splat;
	using std::move;
	STPDeviceRuntimeBinary::STPLoweredName commonName;
	/* -------------------------------- common compiler ----------------------------- */
	{
		//Filename
		constexpr static char CommonGeneratorFilename[] = "./Script/STPCommonGenerator.cu";

		//load source code
		const string commongen_source = STPFile::read(CommonGeneratorFilename);
		//set compiler options
		STPDeviceRuntimeBinary::STPSourceInformation commongen_info = commonSourceInfo;
		//this common generator only contains a few shared variables
		commongen_info.NameExpression
			["STPCommonGenerator::Dimension"]
			["STPCommonGenerator::HalfDimension"]
			["STPCommonGenerator::RenderedDimension"]
			["STPCommonGenerator::Permutation"];
		//compile
		HANDLE_COMPILE(bin_common.compileFromSource("STPCommonGenerator", commongen_source, commongen_info))
		commonName = move(output.LoweredName);
	}
	/* ------------------------------- biomefield compiler ----------------------------- */
	{
		//File name of the generator script
		constexpr static char GeneratorFilename[] = "./Script/STPMultiHeightGenerator.cu";
		constexpr static char BiomePropertyFilename[] = "./STPBiomeProperty.hpp";

		//read script
		const string multiheightfield_source = STPFile::read(GeneratorFilename);
		const string biomeprop_hdr = STPFile::read(BiomePropertyFilename);
		//attach biome property
		STPDeviceRuntimeBinary::STPExternalHeaderSource header;
		header.emplace("STPBiomeProperty", biomeprop_hdr);
		//attach source code and load up default compiler options, it returns a copy
		STPDeviceRuntimeBinary::STPSourceInformation multiheightfield_info = commonSourceInfo;
		//we only need to adjust options that are unique to different sources
		multiheightfield_info.NameExpression
			//global function
			["generateMultiBiomeHeightmap"]
			//constant
			["BiomeTable"];
		//options are all set
		multiheightfield_info.ExternalHeader
			["STPBiomeProperty"];
		HANDLE_COMPILE(bin_biome.compileFromSource(
			"STPMultiHeightGenerator", multiheightfield_source, multiheightfield_info, header))
		this->BiomefieldName = move(output.LoweredName);
	}
	/* ---------------------------------- splatmap compiler -------------------------------- */
	{
		//File name of the generator script
		constexpr static char GeneratorFilename[] = "./Script/STPSplatmapGenerator.cu";

		//read source code
		const string splatmap_source = STPFile::read(GeneratorFilename);
		//load default compiler settings
		STPDeviceRuntimeBinary::STPSourceInformation source_info = commonSourceInfo;
		source_info.NameExpression
			["SplatDatabase"]
			//global function
			["generateTextureSplatmap"];
		//compile
		HANDLE_COMPILE(bin_splat.compileFromSource("STPSplatmapGenerator", splatmap_source, source_info))
		this->SplatmapName = move(output.LoweredName);
	}
	/* -------------------------------------- link -------------------------------------- */
	//setup linker options
	using BinT = STPDeviceRuntimeProgram::STPBinaryType;
	const STPDeviceRuntimeProgram::STPLinkerInformation::STPDataJitOption common_data_option;
	STPDeviceRuntimeProgram::STPLinkerInformation linkInfo;
	STPCompilerLog log;

	linkInfo.LinkerOption
#ifndef NDEBUG
		//if debug has turned on, optimisation must be 0 or linker will be crying...
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)0u)
		(CU_JIT_GENERATE_DEBUG_INFO, (void*)1)
#else
		(CU_JIT_OPTIMIZATION_LEVEL, (void*)4u)
#endif
		(CU_JIT_LOG_VERBOSE, (void*)1)
		(CU_JIT_INFO_LOG_BUFFER, log.linker_info_log)
		(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize)
		(CU_JIT_ERROR_LOG_BUFFER, log.linker_error_log)
		(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize);
	
	//setup module options
	linkInfo.ModuleOption
#ifndef NDEBUG
		(CU_JIT_GENERATE_DEBUG_INFO, (void*)1)
#endif
		(CU_JIT_INFO_LOG_BUFFER, log.module_info_log)
		(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize)
		(CU_JIT_ERROR_LOG_BUFFER, log.module_error_log)
		(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)(uintptr_t)STPCompilerLog::LogSize);
	linkInfo.DataOption.emplace_back(&bin_common, BinT::PTX, common_data_option);
	linkInfo.DataOption.emplace_back(&bin_biome, BinT::PTX, common_data_option);
	linkInfo.DataOption.emplace_back(&bin_splat, BinT::PTX, common_data_option);

	linkInfo.ArchiveOption.emplace_back(STPAlgorithm::STPAlgorithmDeviceInfo::DeviceLibrary, common_data_option);

	try {
		this->GeneratorProgram.linkFromBinary(linkInfo);
		cout << log.linker_info_log << endl;
		cout << log.module_info_log << endl;

		//setup some variables
		CUdeviceptr dimension, half_dimension, rendered_dimension, perm;
		size_t dimensionSize, half_dimensionSize, rendered_dimensionSize, permSize;
		//source information
		const CUmodule program = *this->GeneratorProgram;
		STP_CHECK_CUDA(cuModuleGetGlobal(&dimension, &dimensionSize, program,
			commonName.at("STPCommonGenerator::Dimension").c_str()));
		STP_CHECK_CUDA(cuModuleGetGlobal(&half_dimension, &half_dimensionSize,program,
			commonName.at("STPCommonGenerator::HalfDimension").c_str()));
		STP_CHECK_CUDA(cuModuleGetGlobal(&rendered_dimension, &rendered_dimensionSize, program,
			commonName.at("STPCommonGenerator::RenderedDimension").c_str()));
		STP_CHECK_CUDA(cuModuleGetGlobal(&perm, &permSize, program,
			commonName.at("STPCommonGenerator::Permutation").c_str()));
		//send data
		const vec2 halfDim = static_cast<vec2>(this->Dimension) / 2.0f;
		const uvec2 RenderedDim = this->RenderingRange * this->Dimension;
		STP_CHECK_CUDA(cuMemcpyHtoD(dimension, value_ptr(this->Dimension), dimensionSize));
		STP_CHECK_CUDA(cuMemcpyHtoD(half_dimension, value_ptr(halfDim), half_dimensionSize));
		STP_CHECK_CUDA(cuMemcpyHtoD(rendered_dimension, value_ptr(RenderedDim), rendered_dimensionSize));
		//note that we are copying permutation to device, the underlying pointers are managed by this class
		STP_CHECK_CUDA(cuMemcpyHtoD(perm, &(*this->SimplexPermutation), permSize));
	} catch (const SuperTerrainPlus::STPException::STPCUDAError& error) {
		cerr << error.what() << std::endl;
		cerr << log.linker_error_log << endl;
		cerr << log.module_error_log << endl;
		std::terminate();
	}
}

CUmodule STPCommonCompiler::getProgram() const {
	return *this->GeneratorProgram;
}

const STPDeviceRuntimeBinary::STPLoweredName& STPCommonCompiler::getBiomefieldName() const {
	return this->BiomefieldName;
}

const STPDeviceRuntimeBinary::STPLoweredName& STPCommonCompiler::getSplatmapName() const {
	return this->SplatmapName;
}