#include "STPSplatmapGenerator.h"

#include <iostream>
//Error
#include <SuperTerrain+/Utility/Exception/STPCUDAError.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace STPDemo;
using namespace SuperTerrainPlus::STPDiversity;

//File name of the generator script
constexpr static char GeneratorFilename[] = "./Script/STPSplatmapGenerator.cu";

using std::string;

using std::cout;
using std::cerr;
using std::endl;

using glm::uvec2;
using glm::uvec3;
using glm::value_ptr;

STPSplatmapGenerator::STPSplatmapGenerator
	(const STPTextureDatabase::STPDatabaseView& database_view, const SuperTerrainPlus::STPEnvironment::STPChunkSetting& chunk_setting) :
	STPTextureFactory(database_view, chunk_setting), STPCommonCompiler() {
	//compile source code
	this->initGenerator();
}

void STPSplatmapGenerator::initGenerator() {
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

	//link
	//load default linker settings
	STPCommonCompiler::STPCompilerLog log;
	STPCommonCompiler::STPLinkerInformation linker_info = this->getLinkerOptions(log);
	try {
		this->linkProgram(linker_info, CU_JIT_INPUT_PTX);
		cout << log.linker_info_log << endl;
		cout << log.module_info_log << endl;
	}
	catch (const SuperTerrainPlus::STPException::STPCUDAError& error) {
		cerr << error.what() << std::endl;
		cerr << log.linker_error_log << endl;
		cerr << log.module_error_log << endl;
		std::terminate();
	}

	//copy memory to the program
	CUmodule program = this->getGeneratorModule();
	CUdeviceptr splat_database, mapDim, totalmapDim;
	size_t splat_databaseSize, mapDimSize, totalmapDimSize;
	//get variable names
	const auto& name = this->retrieveSourceLoweredName("STPSplatmapGenerator");
	STPcudaCheckErr(cuModuleGetFunction(&this->SplatmapEntry, program, name.at("generateTextureSplatmap")));
	STPcudaCheckErr(cuModuleGetGlobal(&splat_database, &splat_databaseSize, program, name.at("SplatDatabase")));
	STPcudaCheckErr(cuModuleGetGlobal(&mapDim, &mapDimSize, program, name.at("MapDimension")));
	STPcudaCheckErr(cuModuleGetGlobal(&totalmapDim, &totalmapDimSize, program, name.at("TotalBufferDimension")));
	//copy to device variables
	STPcudaCheckErr(cuMemcpyHtoD(mapDim, value_ptr(this->MapDimension), mapDimSize));
	const uvec2 totalBufferSize = this->MapDimension * this->RenderedChunk;
	STPcudaCheckErr(cuMemcpyHtoD(totalmapDim, value_ptr(totalBufferSize), totalmapDimSize));
}

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;

void STPSplatmapGenerator::splat
	(cudaTextureObject_t biomemap_tex, cudaTextureObject_t heightmap_tex, cudaSurfaceObject_t splatmap_surf, const STPTI::STPSplatGeneratorInformation& info, cudaStream_t stream) const {
	int Mingridsize, blocksize;
	//smart launch config
	STPcudaCheckErr(cuOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, this->SplatmapEntry, nullptr, 0ull, 0));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u);
	const uvec3 Dimgridsize = uvec3((this->MapDimension + Dimblocksize - 1u) / Dimblocksize, info.LocalCount);

	//launch kernel
	size_t buffer_size = 40ull;
	unsigned char buffer[40];
	unsigned char* current_buffer = buffer;

	memcpy(current_buffer, &biomemap_tex, sizeof(biomemap_tex));
	current_buffer += sizeof(biomemap_tex);
	memcpy(current_buffer, &heightmap_tex, sizeof(heightmap_tex));
	current_buffer += sizeof(heightmap_tex);
	memcpy(current_buffer, &splatmap_surf, sizeof(splatmap_surf));
	current_buffer += sizeof(splatmap_surf);
	memcpy(current_buffer, &info, sizeof(STPTI::STPSplatGeneratorInformation));

	void* config[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &buffer_size,
		CU_LAUNCH_PARAM_END
	};
	STPcudaCheckErr(cuLaunchKernel(this->SplatmapEntry,
		Dimgridsize.x, Dimgridsize.y, 1u,
		Dimblocksize.x, Dimblocksize.y, Dimgridsize.z,
		0u, stream, nullptr, config));
	STPcudaCheckErr(cudaGetLastError());
}