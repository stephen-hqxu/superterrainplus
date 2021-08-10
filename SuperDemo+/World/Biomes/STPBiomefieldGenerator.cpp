#pragma once
#include "STPBiomefieldGenerator.h"
#include <STPAlgorithmDeviceInfoDebug.h>
//Error
#include <SuperError+/STPDeviceErrorHandler.h>

//Biome
#include "STPBiomeRegistry.h"

//System
#include <iostream>

using namespace STPDemo;
using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;

using std::string;

//File name of the generator script
constexpr static char GeneratorFilename[] = "./STPMultiHeightGenerator.rtc";
constexpr static char BiomePropertyFilename[] = "./STPBiomeProperty.hpp";
const static string device_include = "-I " + string(SuperTerrainPlus::SuperAlgorithmPlus_DeviceInclude),
	stppermutation_size = "-DPERMUTATION_SIZE=" + std::to_string(sizeof(SuperTerrainPlus::STPCompute::STPPermutation));

STPBiomefieldGenerator::STPBiomefieldGenerator(STPSimplexNoiseSetting& simplex_setting, uvec2 dimension)
	: STPDiversityGeneratorRTC(), Noise_Setting(simplex_setting), MapSize(make_uint2(dimension.x, dimension.y)), Simplex_Permutation(this->Noise_Setting) {
	STPcudaCheckErr(cuCtxGetCurrent(&this->cudaCtx));
	//init our device generator
	//our heightfield setting only available in OCEAN biome for now
	this->initGenerator();
}

void STPBiomefieldGenerator::initGenerator() {
	//read script
	const string multiheightfield_source = this->readSource(GeneratorFilename);
	const string biomeprop_hdr = this->readSource(BiomePropertyFilename);
	//attach biome property
	this->attachHeader("STPBiomeProperty", biomeprop_hdr);
	//attach device algorithm library
	this->attachArchive("SuperAlgorithm+Device", SuperTerrainPlus::SuperAlgorithmPlus_DeviceLibrary);
	//attach source code
	STPDiversityGeneratorRTC::STPSourceInformation multiheightfield_info;
	multiheightfield_info.NameExpression
		//global function
		["generateMultiBiomeHeightmap"]
		//constant
		["BiomeTable"]
		["Permutation"]
		["Dimension"]
		["HalfDimension"];
	multiheightfield_info.Option
		["-std=c++17"]
		["-fmad=false"]
		["-arch=compute_75"]
		["-rdc=true"]
#ifdef _DEBUG
		["-G"]
		["-lineinfo"]
#endif
		["-maxrregcount=80"]
		[stppermutation_size.c_str()]
		//include path
		[device_include.c_str()];
	multiheightfield_info.ExternalHeader
		["STPBiomeProperty"];
	try {
		const string log = this->compileSource("STPMultiHeightGenerator", multiheightfield_source, multiheightfield_info);
		std::cout << log << std::endl;
	}
	catch (const string& error) {
		std::cerr << error << std::endl;
		exit(-1);
	}
	//link
	//linker log output
	constexpr static unsigned int LinkerLogSize = 1024u;
	char linker_info_log[LinkerLogSize], linker_error_log[LinkerLogSize];
	char module_info_log[LinkerLogSize], module_error_log[LinkerLogSize];
	STPDiversityGeneratorRTC::STPLinkerInformation link_info;
	link_info
		.setLinkerOption(CU_JIT_INFO_LOG_BUFFER, linker_info_log)
		.setLinkerOption(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)LinkerLogSize)
		.setLinkerOption(CU_JIT_ERROR_LOG_BUFFER, linker_error_log)
		.setLinkerOption(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)LinkerLogSize)
#ifdef _DEBUG
		//if debug has turned on, optimisation must be 0 or linker will be crying...
		.setLinkerOption(CU_JIT_OPTIMIZATION_LEVEL, (void*)0u)
#else
		.setLinkerOption(CU_JIT_OPTIMIZATION_LEVEL, (void*)3u)
#endif
		.setLinkerOption(CU_JIT_LOG_VERBOSE, (void*)1)
		//no need to generate debug info anymore since our library and runtime script all contain that
#ifdef _DEBUG
		.setModuleLoadOption(CU_JIT_GENERATE_DEBUG_INFO, (void*)1)
#endif
		.setModuleLoadOption(CU_JIT_INFO_LOG_BUFFER, module_info_log)
		.setModuleLoadOption(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, (void*)LinkerLogSize)
		.setModuleLoadOption(CU_JIT_ERROR_LOG_BUFFER, module_error_log)
		.setModuleLoadOption(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, (void*)LinkerLogSize);
	try {
		this->linkProgram(link_info, CU_JIT_INPUT_PTX);
		std::cout << linker_info_log << std::endl;
		std::cout << module_info_log << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << linker_error_log << std::endl;
		std::cerr << module_error_log << std::endl;
		exit(-1);
	}
	
	//global pointers
	CUmodule program = this->getGeneratorModule();
	CUdeviceptr biome_prop, dimension, half_dimension, permutation;
	size_t biome_propSize, dimensionSize, half_dimensionSize, permutationSize;
	//get names and start copying
	const auto& name = this->retrieveSourceLoweredName("STPMultiHeightGenerator");
	STPcudaCheckErr(cuModuleGetFunction(&this->GeneratorEntry, program, name.at("generateMultiBiomeHeightmap")));
	STPcudaCheckErr(cuModuleGetGlobal(&biome_prop, &biome_propSize, program, name.at("BiomeTable")));
	STPcudaCheckErr(cuModuleGetGlobal(&dimension, &dimensionSize, program, name.at("Dimension")));
	STPcudaCheckErr(cuModuleGetGlobal(&half_dimension, &half_dimensionSize, program, name.at("HalfDimension")));
	STPcudaCheckErr(cuModuleGetGlobal(&permutation, &permutationSize, program, name.at("Permutation")));
	//copy variables
	const float2 halfSize = make_float2(this->MapSize.x * 0.5f, this->MapSize.y * 0.5f);
	STPcudaCheckErr(cuMemcpyHtoD(dimension, &this->MapSize, dimensionSize));
	STPcudaCheckErr(cuMemcpyHtoD(half_dimension, &halfSize, half_dimensionSize));
	//note that we are copying permutation to device, the underlying pointers are managed by this class
	STPcudaCheckErr(cuMemcpyHtoD(permutation, &this->Simplex_Permutation(), permutationSize));

	//copy biome properties
	//currently we have two biomes
	STPBiomeProperty* biomeTable_buffer;
	STPcudaCheckErr(cuMemAllocHost(reinterpret_cast<void**>(&biomeTable_buffer), biome_propSize));
	constexpr size_t one_biomeprop_size = sizeof(STPBiomeProperty);
	//copy to host buffer
	memcpy(biomeTable_buffer, dynamic_cast<const STPBiomeProperty*>(&STPBiomeRegistry::OCEAN.getProperties()), one_biomeprop_size);
	memcpy(biomeTable_buffer + 1, dynamic_cast<const STPBiomeProperty*>(&STPBiomeRegistry::PLAINS.getProperties()), one_biomeprop_size);
	//copy everything to device
	STPcudaCheckErr(cuMemcpyHtoD(biome_prop, biomeTable_buffer, biome_propSize));

	STPcudaCheckErr(cuMemFreeHost(biomeTable_buffer));
}
	
void STPBiomefieldGenerator::operator()(float* heightmap, const Sample* biomemap, float2 offset, cudaStream_t stream) const {
	int Mingridsize, blocksize;
	dim3 Dimgridsize, Dimblocksize;
	//smart launch config
	STPcudaCheckErr(cuOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, this->GeneratorEntry, nullptr, 0ull, 0));
	Dimblocksize = dim3(32, blocksize / 32);
	//under-sampled heightmap, and super-sample it back with interpolation
	Dimgridsize = dim3((this->MapSize.x + Dimblocksize.x - 1) / Dimblocksize.x, 
		(this->MapSize.y + Dimblocksize.y - 1) / Dimblocksize.y);

	//launch kernel
	size_t bufferSize = 24ull;
	unsigned char buffer[24];
	memcpy(buffer, &heightmap, sizeof(heightmap));
	memcpy(buffer + 8, &biomemap, sizeof(biomemap));
	memcpy(buffer + 16, &offset, sizeof(offset));
		
	void* config[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &bufferSize,
		CU_LAUNCH_PARAM_END
	};
	STPcudaCheckErr(cuLaunchKernel(this->GeneratorEntry,
		Dimgridsize.x, Dimgridsize.y, Dimgridsize.z,
		Dimblocksize.x, Dimblocksize.y, Dimblocksize.z,
		0u, stream, nullptr, config));
	STPcudaCheckErr(cudaGetLastError());
}