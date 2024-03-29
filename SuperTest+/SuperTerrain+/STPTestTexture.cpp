//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_container_properties.hpp>

//SuperTerrain+/SuperTerrain+/World/Diversity/Texture
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Exception
#include <SuperTerrain+/Exception/API/STPSQLError.h>

#include <algorithm>

namespace Err = SuperTerrainPlus::STPException;
namespace TexInf = SuperTerrainPlus::STPDiversity::STPTextureInformation;

using SuperTerrainPlus::STPDiversity::STPTextureType, SuperTerrainPlus::STPDiversity::STPTextureDatabase;

using glm::uvec2;

using Catch::Matchers::SizeIs;

bool operator==(const STPTextureDatabase::STPMapGroupDescription& v1, const STPTextureDatabase::STPMapGroupDescription& v2) {
	return v1.Dimension == v2.Dimension && 
		v1.MipMapLevel == v2.MipMapLevel &&
		v1.ChannelFormat == v2.ChannelFormat && 
		v1.InteralFormat == v2.InteralFormat &&
		v1.PixelFormat == v2.PixelFormat;
}

bool operator==(const STPTextureDatabase::STPViewGroupDescription& v1, const STPTextureDatabase::STPViewGroupDescription& v2) {
	return v1.PrimaryScale == v2.PrimaryScale &&
		v1.SecondaryScale == v2.SecondaryScale &&
		v1.TertiaryScale == v2.TertiaryScale;
}

SCENARIO_METHOD(STPTextureDatabase, "STPTextureDatabase can store texture information and retrieve whenever needed",
	"[Diversity][Texture][STPTextureDatabase]") {
	auto& Splat = this->splatBuilder();

	GIVEN("A texture database") {
		static constexpr unsigned char DummyTexture[] = { 0u, 1u, 2u, 3u };

		WHEN("The database is freshly created") {

			THEN("Database should have zero size") {
				REQUIRE(this->mapGroupSize() == 0u);
				REQUIRE(this->viewGroupSize() == 0u);
				REQUIRE(this->mapSize() == 0u);
				REQUIRE(this->textureSize() == 0u);
				//internal components in the database
				REQUIRE(Splat.altitudeSize() == 0u);
				REQUIRE(Splat.gradientSize() == 0u);
			}

			THEN("Deletion of members have no effect (by design) on the database") {
				REQUIRE_NOTHROW(this->removeTexture(0u));
				REQUIRE_NOTHROW(this->removeMapGroup(0u));
				REQUIRE_NOTHROW(this->removeViewGroup(0u));
			}

			AND_WHEN("Trying to retrieve data that does not exist in the database") {

				THEN("Operation is halted and error is reported") {
					REQUIRE_THROWS_AS(this->getMapGroupDescription(123u), SuperTerrainPlus::STPException::STPSQLError);
					REQUIRE_THROWS_AS(this->getViewGroupDescription(456u), SuperTerrainPlus::STPException::STPSQLError);
				}

			}
		}

		WHEN("Some texture containers are inserted") {
			static constexpr STPTextureDatabase::STPMapGroupDescription MapDescription = {
				uvec2(2u),
				2u,
				//for simplicity we just mimic some random values for GL constants
				0u,
				1u,
				2u
			};
			static constexpr STPTextureDatabase::STPViewGroupDescription ViewDescription = {
				8u,
				4u,
				2u
			};
			const auto DummyMapGroup = this->addMapGroup(MapDescription);
			const auto DummyViewGroup = this->addViewGroup(ViewDescription);
			const auto DummyTex = this->addTexture(DummyViewGroup);

			THEN("Container can inserted by verifying the number of each member in the database") {
				REQUIRE(this->textureSize() == 1u);
				REQUIRE(this->mapGroupSize() == 1u);
				REQUIRE(this->viewGroupSize() == 1u);

				AND_THEN("Container and container info can be retrieved and the same data is returned") {
					//group desc
					REQUIRE((this->getMapGroupDescription(DummyMapGroup) == MapDescription));
					REQUIRE((this->getViewGroupDescription(DummyViewGroup) == ViewDescription));

					AND_THEN("Container can erased from the database by verifying the size") {
						const auto Path = GENERATE(range(0u, 2u));
						switch (Path) {
						case 0u://remove non-dependent containers
							this->removeTexture(DummyTex);
							this->removeMapGroup(DummyMapGroup);
							this->removeViewGroup(DummyViewGroup);

							REQUIRE(this->textureSize() == 0u);
							REQUIRE(this->mapGroupSize() == 0u);
							REQUIRE(this->viewGroupSize() == 0u);
							break;
						case 1u://remove dependent container
							this->removeViewGroup(DummyViewGroup);

							REQUIRE(this->textureSize() == 0u);
							REQUIRE(this->mapGroupSize() == 1u);
							REQUIRE(this->viewGroupSize() == 0u);
							break;
						default:
							break;
						}
					}
				}
			}

			AND_WHEN("Some texture maps are added into the container") {

				THEN("Map should not be inserted if the containers are invalid") {
					REQUIRE_THROWS_AS(this->addMap(6666666u, STPTextureType::Albedo, 66666666u, DummyTexture), Err::STPSQLError);
					REQUIRE(this->mapSize() == 0u);
				}

				THEN("Map can be inserted into the container by verifying the number of them") {
					REQUIRE_NOTHROW(this->addMap(DummyTex, STPTextureType::Albedo, DummyMapGroup, DummyTexture));
					REQUIRE(this->mapSize() == 1u);

					AND_THEN("Map should be erased if dependent container(s) is/are removed") {
						const auto Path = GENERATE(range(0u, 4u));
						//removal of any dependent container should delete the map
						switch (Path) {
						case 0u: this->removeMapGroup(DummyMapGroup);
							break;
						case 1u: this->removeTexture(DummyTex);
							break;
						case 2u:
							this->removeTexture(DummyTex);
							this->removeMapGroup(DummyMapGroup);
							break;
						case 3u:
							this->removeViewGroup(DummyViewGroup);
							break;
						default:
							break;
						}
						REQUIRE(this->mapSize() == 0u);
					}

					AND_GIVEN("Some texture splat rules") {

						WHEN("Splat map does not reference a valid texture") {

							THEN("Splat rule is hence considered to be invalid and should not be added") {
								REQUIRE_THROWS_AS(Splat.addAltitude(0u, 0.2f, 666666u), Err::STPSQLError);
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.2f, 0.8f, 0.0f, 1.0f, 666666u), Err::STPSQLError);

								REQUIRE(Splat.altitudeSize() == 0u);
								REQUIRE(Splat.gradientSize() == 0u);
							}

						}

						WHEN("Splat rules violate boundary conditions, e.g., max < min") {
							
							THEN("Splat rule should not be added") {
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.2f, 0.8f, 0.8f, 0.3f, DummyTex), Err::STPSQLError);
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.9f, 0.1f, 0.0f, 1.0f, DummyTex), Err::STPSQLError);

								REQUIRE(Splat.gradientSize() == 0u);
							}

						}

						WHEN("Splat map references a valid texture") {

							THEN("Splat rule should be added") {
								REQUIRE_NOTHROW(Splat.addAltitude(0u, 0.2f, DummyTex));
								REQUIRE_NOTHROW(Splat.addGradient(0u, 0.2f, 0.8f, 0.0f, 1.0f, DummyTex));

								REQUIRE(Splat.altitudeSize() == 1u);
								REQUIRE(Splat.gradientSize() == 1u);

								AND_THEN("Splat rule should be removed if depended texture is erased from the database") {
									this->removeTexture(DummyTex);

									REQUIRE(Splat.altitudeSize() == 0u);
									REQUIRE(Splat.gradientSize() == 0u);

								}
							}
							
						}

					}

				}

			}

		}

	}

	GIVEN("A texture database with a lot of data to be loaded") {

		THEN("Loading all data into the database in batch should be successful") {
			//view group
			TexInf::STPViewGroupID ViewGroup[2];
			static constexpr STPTextureDatabase::STPViewGroupDescription big_scale = {
				64u,
				32u,
				16u
			}, small_scale = {
				16u,
				8u,
				4u
			};
			ViewGroup[0] = this->addViewGroup(big_scale);
			ViewGroup[1] = this->addViewGroup(small_scale);

			//let's create some scenarios
			//we deliberately add data in a random order, so we can verify later if all result sets are ordered correctly
			//we will also be adding some unused texture and group and check if the database filters out unused containers
			//texture
			TexInf::STPTextureID Tex[5];
			Tex[0] = this->addTexture(ViewGroup[0], "grass");
			Tex[1] = this->addTexture(ViewGroup[1]);
			Tex[2] = this->addTexture(ViewGroup[1], "small_grass");
			Tex[3] = this->addTexture(ViewGroup[0], "stone");
			Tex[4] = this->addTexture(ViewGroup[1], "soil");

			//map group
			TexInf::STPMapGroupID MapGroup[5];
			static constexpr STPTextureDatabase::STPMapGroupDescription x2_rgb = {
					uvec2(2u),
					4u,
					0u,
					1u,
					2u
			}, x4_rgb = {
					uvec2(4u),
					4u,
					0u,
					2u,
					3u
			}, x4_r = {
					uvec2(4u),
					2u,
					1u,
					2u,
					3u
			};
			MapGroup[0] = this->addMapGroup(x2_rgb);
			MapGroup[1] = this->addMapGroup(x2_rgb);
			MapGroup[2] = this->addMapGroup(x2_rgb);
			MapGroup[3] = this->addMapGroup(x4_rgb);
			MapGroup[4] = this->addMapGroup(x4_r);

			//map
			static constexpr unsigned char TexGrassColor[4] = { 66u }, TexGrassNormal[4] = { 13u },
				TexStoneColor[4] = { 3u };
			static constexpr unsigned char TexSoilColor[16] = { 6u }, TexSoilNormal[16] = { 11u }, TexSoilSpec[16] = { 133u };
			REQUIRE_NOTHROW(this->addMap(Tex[3],
				STPTextureType::Albedo, MapGroup[2], TexStoneColor
			));
			//this texture will not be used by any rule, by definition texture0, group0 will all be "invalid" and will be removed in the batch result
			REQUIRE_NOTHROW(this->addMap(Tex[0], 
				STPTextureType::Normal, MapGroup[0], TexGrassNormal
			));
			//------------------------------------------------------
			REQUIRE_NOTHROW(this->addMap(Tex[4],
				STPTextureType::Albedo, MapGroup[3], TexSoilColor
			));
			REQUIRE_NOTHROW(this->addMap(Tex[4],
				STPTextureType::Normal, MapGroup[3], TexSoilNormal
			));
			REQUIRE_NOTHROW(this->addMap(Tex[4],
				STPTextureType::Roughness, MapGroup[4], TexSoilSpec
			));
			//-----------------------------------------------------
			REQUIRE_NOTHROW(this->addMap(Tex[2],
				STPTextureType::Albedo, MapGroup[2], TexGrassColor
			));
			REQUIRE_NOTHROW(this->addMap(Tex[2],
				STPTextureType::Normal, MapGroup[2], TexGrassNormal
			));
			//-----------------------------------------------------

			//splat rule
			REQUIRE_NOTHROW(Splat.addGradient(66u,
				0.0f, 0.6f, 0.0f, 0.55f, Tex[2]
			));
			REQUIRE_NOTHROW(Splat.addGradient(66u,
				0.65f, 1.0f, 0.55f, 0.95f, Tex[4]
			));
			//--------------------------------------
			REQUIRE_NOTHROW(Splat.addAltitude(13u,
				0.7f, Tex[2]
			));
			REQUIRE_NOTHROW(Splat.addAltitude(13u,
				1.0f, Tex[4]
			));
			//--------------------------------------
			REQUIRE_NOTHROW(Splat.addAltitude(66u,
				1.0f, Tex[3]
			));
			REQUIRE_NOTHROW(Splat.addAltitude(66u,
				0.6f, Tex[4]
			));

			AND_THEN("The number of data in the database should be consistent with what have been added") {
				REQUIRE(this->mapGroupSize() == 5u);
				REQUIRE(this->viewGroupSize() == 2u);
				REQUIRE(this->mapSize() == 7u);
				REQUIRE(this->textureSize() == 5u);
				//internal components in the database
				REQUIRE(Splat.altitudeSize() == 4u);
				REQUIRE(Splat.gradientSize() == 2u);
			}

			WHEN("The database results are queried in batches") {
				using std::get;
				const auto BatchVisitor = this->visit();

				THEN("Batched result sets are verified to be correct") {
					{
						//get all altitude rules
						const auto AltRec = BatchVisitor.getAltitudes();
						CHECK_THAT(AltRec, SizeIs(4u));
						//check for ordering
						const auto [sample, node] = AltRec[2];
						CHECK(sample == 66u);
						CHECK(node.UpperBound == 0.6f);
						CHECK(node.Reference.DatabaseKey == Tex[4]);
					}
					{
						//get all gradient rules
						const auto GraRec = BatchVisitor.getGradients();
						CHECK_THAT(GraRec, SizeIs(2u));
						//pick some data for checking
						const auto [sample, node] = GraRec[0];
						CHECK(sample == 66u);
						//gradients are only sorted by sample, the rest of the order will be the same as how we inserted
						CHECK(node.Reference.DatabaseKey == Tex[2]);
						CHECK(node.LowerBound == 0.0f);
					}
					{
						//get samples that have been added with rules
						const auto SampleRec = BatchVisitor.getValidSample(2u);
						CHECK_THAT(SampleRec, SizeIs(2u));
						const auto [sample, alt_count, gra_count] = SampleRec[1];
						CHECK(sample == 66u);
						CHECK(alt_count == 2u);
						CHECK(gra_count == 2u);
						//edge case checking
						CHECK(get<2>(SampleRec[0]) == 0u);
					}
					{
						//get group that has any map being used by any valid texture
						const auto GroupRec = BatchVisitor.getValidMapGroup();
						CHECK_THAT(GroupRec, SizeIs(3u));
						const auto& [id, data_count, desc] = GroupRec[0];
						CHECK(id == MapGroup[2]);
						CHECK(data_count == 3u);
						CHECK((desc == x2_rgb));
					}
					{
						//get textures that are referenced by any rule
						const auto TexRec = BatchVisitor.getValidTexture();
						CHECK_THAT(TexRec, SizeIs(3u));
						CHECK(TexRec[0].first == Tex[2]);
						CHECK((TexRec[0].second == small_scale));

						CHECK(TexRec[1].first == Tex[3]);
						CHECK((TexRec[1].second == big_scale));

						CHECK(TexRec[2].first == Tex[4]);
						CHECK((TexRec[2].second == small_scale));
					}
					{
						//get maps that are used by valid texture
						const auto MapRec = BatchVisitor.getValidMap();
						CHECK_THAT(MapRec, SizeIs(6u));
						const auto [group, tex, type, data] = MapRec[3];
						const unsigned char* data_uc = reinterpret_cast<const unsigned char*>(data);
						CHECK(group == MapGroup[3]);
						CHECK(tex == Tex[4]);
						CHECK(type == STPTextureType::Albedo);
						CHECK(std::equal(data_uc, data_uc + sizeof(TexSoilColor), TexSoilColor));
					}
					{
						//get types that are used by any rule
						const auto TypeRec = BatchVisitor.getValidMapType(3u);
						CHECK_THAT(TypeRec, SizeIs(3u));
						CHECK(TypeRec[1] == STPTextureType::Normal);
						CHECK(TypeRec[2] == STPTextureType::Roughness);
					}

				}

			}

		}
		
	}

}