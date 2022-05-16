#include "./Script/STPCommonGenerator.cuh"

using namespace SuperTerrainPlus::STPAlgorithm;

//Those variables are defined in this source file, only
__constant__ uint2 STPCommonGenerator::Dimension[1];
__constant__ float2 STPCommonGenerator::HalfDimension[1];
__constant__ uint2 STPCommonGenerator::RenderedDimension[1];

__constant__ STPPermutation STPCommonGenerator::Permutation[1];