//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

//SuperTerrain+/World/Diversity/Texture
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDefinitionLanguage.h>

//Error
#include <SuperTerrain+/Exception/STPSerialisationError.h>

//System
#include <string>
#include <fstream>
#include <streambuf>

#include <optional>

//This is a TDL with correct syntax
constexpr char TerrainTDL[] = "./TestData/Terrain.tdl";

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using std::string;
using std::ifstream;
using std::istreambuf_iterator;

using std::optional;

/**
 * @brief Read all lines from a file into a formatted string.
 * @param filename The filename to be read from.
 * @return The string containing all lines of a file.
*/
static string readAll(const char* __restrict filename) {
	using std::ios;

	ifstream file(filename);
	if (!file) {
		throw STPException::STPSerialisationError("Unable to open the target file.");
	}
	string str;

	//preallocate memory
	file.seekg(0, ios::end);
	str.reserve(file.tellg());
	file.seekg(0, ios::beg);

	return str.assign(istreambuf_iterator<char>(file), istreambuf_iterator<char>());
}

SCENARIO("TDL interpreter parses a TDL script", "[Diversity][Texture][STPTextureDefinitionLanguage]") {
	optional<const STPTextureDefinitionLanguage> Parser;
	STPTextureDefinitionLanguage::STPTextureVariable TexVar;

	GIVEN("A TDL with correct syntax") {

		WHEN("A texture database needs to be filled with texture splatting rules") {

			THEN("TDL can be parsed from source code") {
				REQUIRE_NOTHROW([&Parser]() { Parser.emplace(readAll(TerrainTDL)); }());

				AND_THEN("Splat rules can be loaded into texture database correctly") {
					STPTextureDatabase Database;
					auto& SplatBuilder = Database.getSplatBuilder();
					const auto View = Database.visit();

					REQUIRE_NOTHROW([&Parser, &TexVar, &Database]() { TexVar = Parser.value()(Database); }());

					//verify correctness of texture
					CHECK(TexVar.size() == 4ull);
					CHECK(Database.textureSize() == TexVar.size());
					CHECK(SplatBuilder.altitudeSize() == 6ull);
					CHECK(SplatBuilder.gradientSize() == 3ull);
					//verify rules
					const auto alt = View.getAltitudes();
					const auto gra = View.getGradients();
					const auto sample = View.getValidSample();
					//pick some random samples for testing
					CHECK(sample.size() == 3ull);
					{
						const auto [id, alt_count, gra_count] = sample.at(1);
						CHECK(id == 101u);
						CHECK(alt_count == 2u);
						CHECK(gra_count == 0u);
					}
					{
						//check altitude rules
						const auto [id, altNode] = alt.at(1);
						CHECK(id == 100u);
						CHECK(altNode.UpperBound == 0.7f);
						CHECK(altNode.Reference.DatabaseKey == TexVar.at("grass"));
					}
					{
						//check gradient rules
						const auto [id, graNode] = gra.at(2);
						CHECK(id == 105u);
						CHECK(graNode.minGradient == 0.2f);
						CHECK(graNode.maxGradient == 0.5f);
						CHECK(graNode.LowerBound == 0.2f);
						CHECK(graNode.UpperBound == 0.55f);
						CHECK(graNode.Reference.DatabaseKey == TexVar.at("gravel"));
					}
				}
			}

		}

	}

	GIVEN("A TDL source with incorrect syntax and semantics") {
		using namespace Catch::Matchers;
		auto tryParse = [&Parser](const char* src) {
			Parser.emplace(src);
		};

		WHEN("There is a syntatic error") {
			constexpr char BrokenTDL1[] = "#texture [x]; \n #rule altitude{0:=(0.4f -> x)";
			constexpr char BrokenTDL2[] = "#texture [x]; \n #rule gradient{0:=(0.1f, 0.2f, 0.3f -> x)}";
			constexpr char BrokenTDL3[] = "#texture [x] \n #rule altitude{0:=(0.4f -> x)}";
			constexpr char BrokenTDL4[] = "#texture [x]; \n #rule gradient{0:=(0.1f, 0.2f, 0.3f, 0.4f, x)}";
			constexpr char BrokenTDL5[] = "#texture [x]; \n #rule altitude{0:=(0.4f -> x)};";
			constexpr char BrokenTDL6[] = "#texture [x]; \n #rule altitude{0:=(0.4f $ x)}";

			THEN("TDL interpreter should report the incorrectness and expected syntax") {
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL1), ContainsSubstring("}"));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL2), ContainsSubstring(","));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL3), ContainsSubstring(";"));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL4), ContainsSubstring("-"));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL5), ContainsSubstring("#"));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL6), ContainsSubstring("$"));
			}

		}

		WHEN("There is a semantic error") {
			constexpr char BrokenTDL1[] = "#texture [x]; \n #rule altitude{0:=(0.15f -> y)}";
			constexpr char BrokenTDL2[] = "#texture [x]; \n #altitude{0:=(0.9f -> x)";

			THEN("TDL interpreter should report the incorrect semantic") {
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL1), ContainsSubstring("y"));
				REQUIRE_THROWS_WITH(tryParse(BrokenTDL2), ContainsSubstring("altitude") && ContainsSubstring("Operation code"));
			}

		}

	}

}