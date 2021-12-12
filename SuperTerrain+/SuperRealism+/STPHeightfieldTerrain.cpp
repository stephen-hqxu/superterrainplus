#include <SuperRealism+/Renderer/STPHeightfieldTerrain.h>
#include <SuperRealism+/STPRealismInfo.h>

//Container
#include <array>

using std::array;

using namespace SuperTerrainPlus::STPRealism;

constexpr static array<signed char, 56ull> PlaneVertex = {
	//Position		//Texcoords		//Normal	//Tangent	//Bitangent
	0, 0, 0,		0, 0,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	1, 0, 0,		1, 0,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	1, 0, 1,		1, 1,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	0, 0, 1,		0, 1,			0, 1, 0,	1, 0, 0,	0, 0, -1
};
constexpr static array<unsigned char, 6ull> PlaneIndex = {
	0, 1, 2,
	0, 2, 3
};

STPHeightfieldTerrain::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline) : TerrainGenerator(generator_pipeline) {

}