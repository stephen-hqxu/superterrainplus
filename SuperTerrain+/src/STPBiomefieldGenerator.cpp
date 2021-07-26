#pragma once
#include <World/Biome/Biomes/STPBiomefieldGenerator.h>

//Generator
#include <World/Biome/Biomes/STPMultiHeightGenerator.cuh>
//Biome
#include <World/Biome/Biomes/STPBiomeRegistry.h>

using namespace STPDemo;
using namespace SuperTerrainPlus::STPSettings;

using glm::uvec2;

STPBiomefieldGenerator::STPBiomefieldGenerator(STPSimplexNoiseSettings& simplex_setting, uvec2 dimension)
	: STPDiversityGenerator(), Noise_Settings(simplex_setting), MapSize(make_uint2(dimension.x, dimension.y)), SimplexNoise(&this->Noise_Settings) {
	//init our device generator
	//our heightfield setting only availabel in OCEAN biome for now
	STPMultiHeightGenerator::initGenerator(&STPBiomeRegistry::OCEAN.getProperties(), &this->SimplexNoise, this->MapSize);
}

void STPBiomefieldGenerator::operator()(float* heightmap, const Sample* biomemap, float2 offset, cudaStream_t stream) const {
	STPMultiHeightGenerator::generateHeightmap(heightmap, this->MapSize, offset, stream);
}