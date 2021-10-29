//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>

//SuperTerrain+/SuperTerrain+/World/Diversity/Texture
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Exception
#include <SuperTerrain+/Utility/Exception/STPDatabaseError.h>
#include <SuperTerrain+/Utility/Exception/STPBadNumericRange.h>

#include <algorithm>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

using glm::uvec2;

bool operator==(const STPTextureDatabase::STPTextureDescription& v1, const STPTextureDatabase::STPTextureDescription& v2) {
	return v1.Dimension == v2.Dimension && 
		v1.ChannelFormat == v2.ChannelFormat && 
		v1.InteralFormat == v2.InteralFormat &&
		v1.PixelFormat == v2.PixelFormat;
}

SCENARIO_METHOD(STPTextureDatabase, "STPTextureDatabase can store texture information and retrieve whenever needed",
	"[Diversity][Texture][STPTextureDatabase]") {
	auto& Splat = this->getSplatBuilder();

	GIVEN("A texture database") {
		static constexpr unsigned char DummyTexture[] = { 0u, 1u, 2u, 3u };

		WHEN("The database is freshly created") {

			THEN("Database should have zero size") {
				REQUIRE(this->groupSize() == 0ull);
				REQUIRE(this->mapSize() == 0ull);
				REQUIRE(this->textureSize() == 0ull);
				//internal components in the database
				REQUIRE(Splat.altitudeSize() == 0ull);
				REQUIRE(Splat.gradientSize() == 0ull);
			}

			THEN("Deletion of members have no effect (by design) on the database") {
				REQUIRE_NOTHROW(this->removeTexture(0u));
				REQUIRE_NOTHROW(this->removeGroup(0u));
			}

			AND_WHEN("Trying to retrieve data that does not exist in the database") {

				THEN("Operation is halted and error is reported") {
					REQUIRE_THROWS_AS(this->getGroupDescription(123u), SuperTerrainPlus::STPException::STPDatabaseError);
				}

			}
		}

		WHEN("Some texture containers are inserted") {
			static constexpr STPTextureDatabase::STPTextureDescription Description = {
				uvec2(2u),
				//for simplicity we just mimic some random values for GL constants
				0u,
				1u,
				2u
			};
			const auto DummyTex = this->addTexture();
			const auto DummyGroup = this->addGroup(Description);

			THEN("Container can inserted by verifying the number of each member in the database") {
				REQUIRE(this->textureSize() == 1ull);
				REQUIRE(this->groupSize() == 1ull);

				AND_THEN("Container and container info can be retrieved and the same data is returned") {
					//group desc
					REQUIRE((this->getGroupDescription(DummyGroup) == Description));

					AND_THEN("Container can erased from the database by verifying the size") {
						this->removeTexture(DummyTex);
						this->removeGroup(DummyGroup);

						REQUIRE(this->textureSize() == 0ull);
						REQUIRE(this->groupSize() == 0ull);
					}
				}
			}

			AND_WHEN("Some texture maps are added into the container") {

				THEN("Map should not be inserted if the containers are invalid") {
					REQUIRE_THROWS_AS(this->addMap(6666666u, STPTextureType::Albedo, 66666666u, DummyTexture), STPException::STPDatabaseError);
					REQUIRE(this->mapSize() == 0u);
				}

				THEN("Map can be inserted into the container by verifying the number of them") {
					REQUIRE_NOTHROW(this->addMap(DummyTex, STPTextureType::Albedo, DummyGroup, DummyTexture));
					REQUIRE(this->mapSize() == 1u);

					AND_THEN("Map should be erased if dependent container(s) is/are removed") {
						const auto Path = GENERATE(range(0u, 3u));
						//removal of any dependent container should delete the map
						switch (Path) {
						case 0u: this->removeGroup(DummyGroup);
							break;
						case 1u: this->removeTexture(DummyTex);
							break;
						case 2u:
							this->removeTexture(DummyTex);
							this->removeGroup(DummyGroup);
							break;
						default:
							break;
						}
						REQUIRE(this->mapSize() == 0u);
					}

					AND_GIVEN("Some texture splat rules") {

						WHEN("Splat map does not reference a valid texture") {

							THEN("Splat rule is hence considered to be invalid and should not be added") {
								REQUIRE_THROWS_AS(Splat.addAltitude(0u, 0.2f, 666666u), STPException::STPDatabaseError);
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.2f, 0.8f, 0.0f, 1.0f, 666666u), STPException::STPDatabaseError);

								REQUIRE(Splat.altitudeSize() == 0ull);
								REQUIRE(Splat.gradientSize() == 0ull);
							}

						}

						WHEN("Splat rules violate boundary conditions, e.g., max < min") {
							
							THEN("Splat rule should not be added") {
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.2f, 0.8f, 0.8f, 0.3f, DummyTex), STPException::STPDatabaseError);
								REQUIRE_THROWS_AS(Splat.addGradient(0u, 0.9f, 0.1f, 0.0f, 1.0f, DummyTex), STPException::STPDatabaseError);

								REQUIRE(Splat.gradientSize() == 0ull);
							}

						}

						WHEN("Splat map references a valid texture") {

							THEN("Splat rule should be added") {
								REQUIRE_NOTHROW(Splat.addAltitude(0u, 0.2f, DummyTex));
								REQUIRE_NOTHROW(Splat.addGradient(0u, 0.2f, 0.8f, 0.0f, 1.0f, DummyTex));

								REQUIRE(Splat.altitudeSize() == 1ull);
								REQUIRE(Splat.gradientSize() == 1ull);

								AND_THEN("Splat rule should be removed if depended texture is erased from the database") {
									this->removeTexture(DummyTex);

									REQUIRE(Splat.altitudeSize() == 0ull);
									REQUIRE(Splat.gradientSize() == 0ull);

								}
							}
							
						}

					}

				}

			}

		}

	}

	GIVEN("A texture database with a lot of data to be loaded") {

		WHEN("Trying to insert zero number of texture into the database") {

			THEN("Insertion should be rejected") {
				STPTextureInformation::STPTextureID BrokenTex;
				REQUIRE_THROWS_AS(this->addTexture(0u, &BrokenTex), STPException::STPBadNumericRange);
			}

		}

		THEN("Loading all data into the database in batch should be successful") {
			//let's create some scenarios
			//we deliberately add data in a random order, so we can verify later if all result sets are ordered correctly
			//we will also be adding some unused texture and group and check if the database filters out unused containers
			//texture
			STPTextureInformation::STPTextureID Tex[5];
			REQUIRE_NOTHROW(this->addTexture(5u, Tex));

			//group
			STPTextureInformation::STPTextureGroupID Group[5];
			static constexpr STPTextureDatabase::STPTextureDescription x2_rgb = {
					uvec2(2u),
					0u,
					1u,
					2u
			}, x4_rgb = {
					uvec2(4u),
					0u,
					2u,
					3u
			}, x4_r = {
					uvec2(4u),
					1u,
					2u,
					3u
			};
			Group[0] = this->addGroup(x2_rgb);
			Group[1] = this->addGroup(x2_rgb);
			Group[2] = this->addGroup(x2_rgb);
			Group[3] = this->addGroup(x4_rgb);
			Group[4] = this->addGroup(x4_r);

			//map
			static constexpr unsigned char TexGrassColor[4] = { 66u }, TexGrassNormal[4] = { 13u },
				TexStoneColor[4] = { 3u };
			static constexpr unsigned char TexSoilColor[16] = { 6u }, TexSoilNormal[16] = { 11u }, TexSoilSpec[16] = { 133u };
			REQUIRE_NOTHROW(this->addMap(Tex[3],
				STPTextureType::Albedo, Group[2], TexStoneColor
			));
			//this texture will not be used by any rule, by definition texture0, group0 will all be "invalid" and will be removed in the batch result
			REQUIRE_NOTHROW(this->addMap(Tex[0], 
				STPTextureType::Normal, Group[0], TexGrassNormal
			));
			REQUIRE_NOTHROW(this->addMaps(Tex[4],
				STPTextureType::Albedo, Group[3], TexSoilColor,
				STPTextureType::Normal, Group[3], TexSoilNormal,
				STPTextureType::Specular, Group[4], TexSoilSpec
			));
			REQUIRE_NOTHROW(this->addMaps(Tex[2],
				STPTextureType::Albedo, Group[2], TexGrassColor,
				STPTextureType::Normal, Group[2], TexGrassNormal
			));

			//splat rule
			REQUIRE_NOTHROW(Splat.addGradients(66u,
				0.0f, 0.6f, 0.0f, 0.55f, Tex[2],
				0.65f, 1.0f, 0.55f, 0.95f, Tex[4]
			));
			REQUIRE_NOTHROW(Splat.addAltitudes(13u,
				0.7f, Tex[2],
				1.0f, Tex[4]
			));
			REQUIRE_NOTHROW(Splat.addAltitudes(66u,
				1.0f, Tex[3],
				0.6f, Tex[4]
			));

			AND_THEN("The number of data in the database should be consistent with what have been added") {
				REQUIRE(this->groupSize() == 5ull);
				REQUIRE(this->mapSize() == 7ull);
				REQUIRE(this->textureSize() == 5ull);
				//internal components in the database
				REQUIRE(Splat.altitudeSize() == 4ull);
				REQUIRE(Splat.gradientSize() == 2ull);
			}

			WHEN("The database results are queried in batches") {
				using std::get;
				const auto BatchVisitor = this->visit();

				AND_WHEN("The query parameters contain errors") {

					THEN("Query should fail and error is reported") {
						REQUIRE_THROWS_AS(BatchVisitor.getValidMapType(666u), STPException::STPBadNumericRange);
					}

				}

				AND_WHEN("Query is valid") {

					THEN("Batched result sets are verified to be correct") {
						{
							//get all altitude rules
							const auto AltRec = BatchVisitor.getAltitudes();
							CHECK(AltRec.size() == 4ull);
							//check for ordering
							const auto [sample, node] = AltRec[2];
							CHECK(sample == 66u);
							CHECK(node.UpperBound == 0.6f);
							CHECK(node.Reference.DatabaseKey == Tex[4]);
						}
						{
							//get all gradient rules
							const auto GraRec = BatchVisitor.getGradients();
							CHECK(GraRec.size() == 2ull);
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
							CHECK(SampleRec.size() == 2ull);
							const auto [sample, alt_count, gra_count] = SampleRec[1];
							CHECK(sample == 66u);
							CHECK(alt_count == 2ull);
							CHECK(gra_count == 2ull);
							//edge case checking
							CHECK(get<2>(SampleRec[0]) == 0ull);
						}
						{
							//get group that has any map being used by any valid texture
							const auto GroupRec = BatchVisitor.getValidGroup();
							CHECK(GroupRec.size() == 3ull);
							const auto& [id, data_count, desc] = GroupRec[0];
							CHECK(id == Group[2]);
							CHECK(data_count == 3ull);
							CHECK((desc == x2_rgb));
						}
						{
							//get textures that are referenced by any rule
							const auto TexRec = BatchVisitor.getValidTexture();
							CHECK(TexRec.size() == 3ull);
							CHECK(TexRec[0] == Tex[2]);
							CHECK(TexRec[1] == Tex[3]);
							CHECK(TexRec[2] == Tex[4]);
						}
						{
							//get maps that are used by valid texture
							const auto MapRec = BatchVisitor.getValidMap();
							CHECK(MapRec.size() == 6ull);
							const auto [group, tex, type, data] = MapRec[3];
							const unsigned char* data_uc = reinterpret_cast<const unsigned char*>(data);
							CHECK(group == Group[3]);
							CHECK(tex == Tex[4]);
							CHECK(type == STPTextureType::Albedo);
							CHECK(std::equal(data_uc, data_uc + sizeof(TexSoilColor), TexSoilColor));
						}
						{
							//get types that are used by any rule
							const auto TypeRec = BatchVisitor.getValidMapType(3u);
							CHECK(TypeRec.size() == 3ull);
							CHECK(TypeRec[1] == STPTextureType::Normal);
							CHECK(TypeRec[2] == STPTextureType::Specular);
						}

					}

				}

			}

		}
		
	}

}