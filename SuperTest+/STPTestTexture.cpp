//Catch2
#include <catch2/catch_test_macros.hpp>

//SuperTerrain+/SuperTerrain+/World/Diversity/Texture
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Exception
#include <SuperTerrain+/Utility/Exception/STPDatabaseError.h>

using namespace SuperTerrainPlus::STPDiversity;

SCENARIO_METHOD(STPTextureDatabase, "STPTextureDatabase can store texture information and retrieve whenever needed",
	"[Diversity][Texture][STPTextureDatabase]") {

	GIVEN("An empty texture database") {

		WHEN("Inserting some containers like texture and group and texture map") {

			SUCCEED("TODO");

		}

	}

}