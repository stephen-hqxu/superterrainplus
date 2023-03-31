#include "STPSplatmapGenerator.h"

#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//GL
#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

using namespace STPDemo;
using namespace SuperTerrainPlus::STPDiversity;

using std::string;

using std::cout;
using std::cerr;
using std::endl;

using glm::uvec2;
using glm::uvec3;
using glm::value_ptr;

STPSplatmapGenerator::STPSplatmapGenerator(const STPCommonCompiler& program,
	const STPTextureDatabase::STPDatabaseView& database_view,
	const SuperTerrainPlus::STPEnvironment::STPChunkSetting& chunk_setting, const float anisotropy) :
	//bias is a scaling value, we need to calculate the reciprocal.
	STPTextureFactory(database_view, chunk_setting, [anisotropy](auto tbo) {
		glTextureParameteri(tbo, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(tbo, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTextureParameteri(tbo, GL_TEXTURE_WRAP_R, GL_REPEAT);
		glTextureParameteri(tbo, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTextureParameteri(tbo, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTextureParameterf(tbo, GL_TEXTURE_MAX_ANISOTROPY, anisotropy);

		glGenerateTextureMipmap(tbo);	
	}),
	KernelProgram(program) {
	//compile source code
	this->initGenerator();
}

void STPSplatmapGenerator::initGenerator() {
	//copy memory to the program
	CUmodule program = this->KernelProgram.getProgram();
	CUdeviceptr splat_database;
	size_t splat_databaseSize;
	//get variable names
	const auto& name = this->KernelProgram.getSplatmapName();
	STP_CHECK_CUDA(cuModuleGetFunction(&this->SplatmapEntry, program, name.at("generateTextureSplatmap").c_str()));
	STP_CHECK_CUDA(cuModuleGetGlobal(&splat_database, &splat_databaseSize, program, name.at("SplatDatabase").c_str()));
	//add splat-database and gradient bias
	const STPTextureInformation::STPSplatRuleDatabase splatDb = this->getSplatDatabase();
	STP_CHECK_CUDA(cuMemcpyHtoD(splat_database, &splatDb, splat_databaseSize));
}

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;

void STPSplatmapGenerator::splat(const cudaTextureObject_t biomemap_tex, const cudaTextureObject_t heightmap_tex,
	const cudaSurfaceObject_t splatmap_surf, const STPTI::STPSplatGeneratorInformation& info, const cudaStream_t stream) const {
	int minGridSize, bestBlockSize;
	//smart launch config
	STP_CHECK_CUDA(cuOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, this->SplatmapEntry, nullptr, 0u, 0));
	(void)minGridSize;
	const auto [gridSize, blockSize] = SuperTerrainPlus::STPDeviceLaunchSetup::determineLaunchConfiguration<2u>(
		bestBlockSize, uvec3(this->MapDimension, info.LocalCount));

	//launch kernel
	constexpr static size_t BufferSize = sizeof(biomemap_tex) + sizeof(heightmap_tex) + sizeof(splatmap_surf)
		+ sizeof(STPTI::STPSplatGeneratorInformation);
	size_t buffer_size = BufferSize;
	unsigned char buffer[BufferSize];
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
	STP_CHECK_CUDA(cuLaunchKernel(this->SplatmapEntry,
		gridSize.x, gridSize.y, gridSize.z,
		blockSize.x, blockSize.y, blockSize.z,
		0u, stream, nullptr, config));
}