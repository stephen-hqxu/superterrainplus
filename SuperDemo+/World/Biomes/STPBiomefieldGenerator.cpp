#pragma once
#include "STPBiomefieldGenerator.h"

//System
#include <fstream>
//Biome
#include "STPBiomeRegistry.h"

//File name of the generator script
constexpr static char GeneratorFilename[] = "./STPMultiHeightGenerator.rtc";

using namespace STPDemo;
using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;

using std::ifstream;

STPBiomefieldGenerator::STPBiomefieldGenerator(STPSimplexNoiseSetting& simplex_setting, uvec2 dimension)
	: STPDiversityGeneratorRTC(), Noise_Setting(simplex_setting), MapSize(make_uint2(dimension.x, dimension.y)), Simplex_Permutation(this->Noise_Setting) {
	//init our device generator
	//our heightfield setting only available in OCEAN biome for now
	this->initGenerator(&STPBiomeRegistry::OCEAN.getProperties());
}

void STPBiomefieldGenerator::initGenerator(const STPBiomeSettings* biome_settings) {
	//read script
	std::ifstream script_reader(GeneratorFilename);
	assert(script_reader);

	
}

void STPBiomefieldGenerator::operator()(float* heightmap, const Sample* biomemap, float2 offset, cudaStream_t stream) const {
	//int Mingridsize, blocksize;
	//dim3 Dimgridsize, Dimblocksize;
	////smart launch config
	//STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &generateMultiBiomeHeightmap));
	//Dimblocksize = dim3(32, blocksize / 32);
	//Dimgridsize = dim3((this->MapSize.x + Dimblocksize.x - 1) / Dimblocksize.x, (this->MapSize.y + Dimblocksize.y - 1) / Dimblocksize.y);
}