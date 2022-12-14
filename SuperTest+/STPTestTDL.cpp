//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

//SuperAlgorithm+/Parser
#include <SuperAlgorithm+/Parser/STPTextureDefinitionLanguage.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

#include <string>
#include <optional>

//This is a TDL with correct syntax
constexpr static char TerrainTDL[] = "./TestData/Terrain.tdl";

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPAlgorithm;
using namespace SuperTerrainPlus::STPDiversity;

using std::string;
using std::optional;

SCENARIO("TDL interpreter parses a TDL script", "[AlgorithmHost][Texture][STPTextureDefinitionLanguage]") {
	GIVEN("A TDL with correct syntax") {

		WHEN("A texture database needs to be filled with texture splatting rules") {

			THEN("TDL can be parsed from source code") {
				const string TestTDLString = STPFile::read(TerrainTDL);
				STPTextureDefinitionLanguage::STPResult Result;
				REQUIRE_NOTHROW([&Result, &TestTDLString]() {
					Result = STPTextureDefinitionLanguage::read(TestTDLString.c_str(), TerrainTDL);
				}());

				AND_THEN("Splat rules can be loaded into texture database correctly") {
					STPTextureDatabase Database;
					auto& SplatBuilder = Database.splatBuilder();
					const auto View = Database.visit();
					STPTextureDefinitionLanguage::STPResult::STPTextureVariable TexVar;

					REQUIRE_NOTHROW([&Result, &TexVar, &Database]() { TexVar = Result.load(Database); }());

					//verify correctness of texture
					CHECK(TexVar.size() == 4u);
					CHECK(Database.textureSize() == TexVar.size());
					CHECK(SplatBuilder.altitudeSize() == 6u);
					CHECK(SplatBuilder.gradientSize() == 3u);
					//verify rules
					const auto alt = View.getAltitudes();
					const auto gra = View.getGradients();
					const auto sample = View.getValidSample();
					//pick some random samples for testing
					CHECK(sample.size() == 3u);
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

						const auto [texture_id, view_id] = TexVar.at("grass");
						CHECK(altNode.Reference.DatabaseKey == texture_id);
						CHECK(Database.getViewGroupDescription(view_id).SecondaryScale == 20u);
					}
					{
						//check gradient rules
						const auto [id, graNode] = gra.at(2);
						CHECK(id == 105u);
						CHECK(graNode.minGradient == 0.2f);
						CHECK(graNode.maxGradient == 0.5f);
						CHECK(graNode.LowerBound == 0.2f);
						CHECK(graNode.UpperBound == 0.55f);

						const auto [texture_id, view_id] = TexVar.at("gravel");
						CHECK(graNode.Reference.DatabaseKey == texture_id);
						CHECK(Database.getViewGroupDescription(view_id).PrimaryScale == 40u);
					}
				}
			}

		}

	}

	GIVEN("A TDL source with incorrect syntax and semantics") {
		using namespace Catch::Matchers;
		const auto tryParse = [](const char* const src) {
			STPTextureDefinitionLanguage::read(src, "BrokenTDLTest");
		};

		WHEN("There is a syntactic error") {
			constexpr char BrokenTDL1[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule altitude{0:=(0.4f -> x)";
			constexpr char BrokenTDL2[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule gradient{0:=(0.1f, 0.2f, 0.3f -> x)}";
			constexpr char BrokenTDL3[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule gradient{0:=(0.1f, 0.2f, 0.3f, 0.4f, x)}";
			constexpr char BrokenTDL4[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule altitude{0:=(0.4f -> x)};";
			constexpr char BrokenTDL5[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule altitude{0:=(0.4f $ x)}";
			constexpr char BrokenTDL6[] = "#texture [x] #group view{x:=(1u,2u,3u} \n #rule altitude{0:=(0.4f -> x)}";
			constexpr char BrokenTDL7[] = "#texture [x] #group view{x:=(1u,2u,3u)} \n #rule altitude{0:=(888888888888888888888888888888888888888.8f -> x)}";

			THEN("TDL interpreter should report the mistakes and expected syntax") {
				CHECK_THROWS_WITH(tryParse(BrokenTDL1), ContainsSubstring("}"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL2), ContainsSubstring(","));
				CHECK_THROWS_WITH(tryParse(BrokenTDL3), ContainsSubstring("-"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL4), ContainsSubstring("#"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL5), ContainsSubstring("InvalidToken"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL6), ContainsSubstring(")"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL7), ContainsSubstring("8.8f"));
			}

		}

		WHEN("There is a semantic error") {
			constexpr char BrokenTDL1[] = "#texture [x] #group view{x:=(3u,2u,1u)} \n #rule altitude{0:=(0.15f -> y)}";
			constexpr char BrokenTDL2[] = "#texture [x] #group view{x:=(3u,2u,1u)} \n #altitude{0:=(0.9f -> x)";
			constexpr char BrokenTDL3[] = "#texture [x] #group hey{x:=(3u,2u,1u)} \n #rule altitude{0:=(0.5f -> x)";
			constexpr char BrokenTDL4[] = "#texture [x, y] #group view{x:=(3u,2u,1u)} \n #rule altitude{0:=(0.25f -> x)";

			THEN("TDL interpreter should report the incorrect semantic") {
				CHECK_THROWS_WITH(tryParse(BrokenTDL1), ContainsSubstring("y"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL2), ContainsSubstring("altitude") && ContainsSubstring("Directive"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL3), ContainsSubstring("hey") && ContainsSubstring("Group type"));
				CHECK_THROWS_WITH(tryParse(BrokenTDL4), ContainsSubstring("y"));
			}

		}

	}

}