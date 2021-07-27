#pragma once
#include "STPBiomefieldGenerator.h"

//Generator
#include "STPMultiHeightGenerator.cuh"
//Biome
#include "STPBiomeRegistry.h"

using namespace STPDemo;
using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;

STPBiomefieldGenerator::STPBiomefieldGenerator(STPSimplexNoiseSetting& simplex_setting, uvec2 dimension)
	: STPDiversityGenerator(), Noise_Setting(simplex_setting), MapSize(make_uint2(dimension.x, dimension.y)), SimplexNoise(&this->Noise_Setting) {
	//init our device generator
	//our heightfield setting only availabel in OCEAN biome for now
	STPMultiHeightGenerator::initGenerator(&STPBiomeRegistry::OCEAN.getProperties(), &this->SimplexNoise, this->MapSize);
}

void STPBiomefieldGenerator::operator()(float* heightmap, const Sample* biomemap, float2 offset, cudaStream_t stream) const {
	STPMultiHeightGenerator::generateHeightmap(heightmap, this->MapSize, offset, stream);
}