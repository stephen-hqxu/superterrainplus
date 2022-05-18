//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/SuperTerrain+/World/Diversity
#include <SuperTerrain+/World/Diversity/STPLayerCache.h>
#include <SuperTerrain+/World/Diversity/STPLayerManager.h>
#include <SuperTerrain+/World/Diversity/STPBiomeFactory.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

#include <glm/gtc/type_ptr.hpp>

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;
using SuperTerrainPlus::STPDiversity::Sample;
using SuperTerrainPlus::STPDiversity::Seed;

using glm::ivec2;
using glm::uvec2;
using glm::ivec3;
using glm::uvec3;

class LayerCacheTester : protected STPLayerCache {
protected:

	constexpr static size_t CacheSize = 32ull;

public:

	LayerCacheTester() : STPLayerCache(LayerCacheTester::CacheSize) {

	}

};

SCENARIO_METHOD(LayerCacheTester, "STPLayerCache is used to cache data", "[Diversity][STPLayerCache]") {

	GIVEN("A layer cache with wrong cache size") {
		using namespace SuperTerrainPlus;

		THEN("Layer cache should not be created") {
			const size_t WrongSize = GENERATE(take(3, filter([](size_t i) { return i % 2ull == 1ull; }, random(0ull, 131313131ull))));
			//simple test
			REQUIRE_THROWS_AS(STPLayerCache(WrongSize), STPException::STPBadNumericRange);
		}

		//edge case
		REQUIRE_THROWS_AS(STPLayerCache(0ull), STPException::STPBadNumericRange);
	}

	GIVEN("A valid layer cache") {

		THEN("Cache size can be retrieved") {
			REQUIRE(this->getCapacity() == LayerCacheTester::CacheSize);
		}

		WHEN("Using cache to store and read value back") {
			const auto Coordinate = GENERATE(take(3, chunk(3, random(-987654, 1313666))));

			THEN("A new cache should have no entry within") {
				const STPCacheEntry NewEntry = this->locate(Coordinate[0], Coordinate[1], Coordinate[2]);
				const STPCacheData NewData = this->read(NewEntry);

				REQUIRE_FALSE(NewData.has_value());

				AND_THEN("Data can be retrieved when it has been cached") {
					const Sample Written = static_cast<Sample>(Coordinate[0] + Coordinate[1] + Coordinate[2]);
					this->write(NewEntry, Written);
					//locate the entry again
					const STPCacheEntry AgainEntry = this->locate(Coordinate[0], Coordinate[1], Coordinate[2]);
					const STPCacheData AgainData = this->read(AgainEntry);

					REQUIRE(AgainData.has_value());
					REQUIRE(*AgainData == Written);
				}

			}

		}

	}

}

class RootLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		//unsigned type wins!!!
		return static_cast<Sample>(1u + x + y + z);
	}

public:

	RootLayer(Seed seed, Seed salt) : STPLayer(seed, salt) {

	}

};

class NormalLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		//simply use the value from the parent
		return this->getAscendant()->retrieve(x, y, z);
	}

public:

	NormalLayer(Seed seed, Seed salt, STPLayer* ascendant) : STPLayer(seed, salt, ascendant) {

	}

};

class MergingLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		return static_cast<Sample>(this->getAscendant(0)->retrieve(x, y, z) + this->getAscendant(1)->retrieve(x, y, z));
	}

public:

	MergingLayer(Seed seed, Seed salt, STPLayer* asc1, STPLayer* asc2) : STPLayer(seed, salt, asc1, asc2) {

	}

};

class RandomLayer : public STPLayer {
private:

	Sample sample(int x, int, int z) {
		return static_cast<Sample>(this->genLocalSeed(x, z));
	}

protected:

	inline Sample getValue(ivec2 coord) {
		return this->sample(coord.x, 0, coord.y);
	}

	inline ivec2 getRandomCoord() const {
		const auto RandomCoord = GENERATE(take(2, chunk(2, random(-666666, 666666))));
		return glm::make_vec2(RandomCoord.data());
	}

public:

	RandomLayer() : STPLayer(19657483ull, 48321567ull) {
		
	}

};

SCENARIO_METHOD(RandomLayer, "STPLayer generates random seed with built-in RNG", "[Diversity][STPLayer][!mayfail]") {

	GIVEN("A random seed to be mixed with another seed") {
		const auto SeedSalt = GENERATE(take(3, chunk(2, random(-123456789876, 66666666666))));
		const Seed OldSeed = static_cast<unsigned long long>(SeedSalt[0]);

		WHEN("Another seed needs to be generated from the original seed") {

			THEN("The two seeds are distinct") {
				//two seeds might be the same (very unlikely)
				const Seed NewSeed = RandomLayer::mixSeed(OldSeed, SeedSalt[1]);
				const bool SeedEqual = NewSeed == OldSeed;

				CHECK_FALSE(SeedEqual);
				CHECKED_IF(SeedEqual) {
					WARN("Seeds are equal, possibly due to hash collision, run again to confirm.");
				}

				AND_THEN("The same input seed should produce the same output seed") {
					CHECK(RandomLayer::mixSeed(OldSeed, SeedSalt[1]) == NewSeed);
				}
			}

		}

	}

	GIVEN("An implementation of STPLayer") {
		const ivec2 Coord = this->getRandomCoord();

		WHEN("Local seed is retrieved from the layer") {
			const Sample Value1 = this->getValue(Coord);

			THEN("Seed should be deterministic when the input is the same") {
				const Sample Value2 = this->getValue(Coord);
				CHECK(Value2 == Value1);
			}

			THEN("Seed should be random when the inputs are different") {
				const ivec2 NewCoord = this->getRandomCoord();
				const Sample Value2 = this->getValue(NewCoord);

				const bool result = Value2 == Value1;
				CHECKED_IF(result) {
					WARN("Assertion involves comparison of random values and it might fail due to RNG, run the test again with a different seed value.");
				}
				CHECK_FALSE(result);
			}

		}

		WHEN("Asking for a local random number generator from the layer") {
			const Seed LocalSeed = this->genLocalSeed(Coord.x, Coord.y);

			THEN("The RNG loaded with the same local seed should produce the same sequence of output") {
				constexpr static unsigned int BucketSize = 16u;
				constexpr static Sample Bound = std::numeric_limits<Sample>::max();

				const STPLayer::STPLocalRNG RNG1 = this->getRNG(LocalSeed),
					RNG2 = this->getRNG(LocalSeed);

				bool equalNextval = true, equalChoose = true;
				for (unsigned int i = 0u; i < BucketSize && (equalNextval || equalChoose); i++) {
					equalNextval &= RNG1.nextVal(Bound) == RNG2.nextVal(Bound);
					equalChoose &= RNG1.choose(0u, 1u, 2u, 3u) == RNG2.choose(0u, 1u, 2u, 3u);
				}
				CHECK(equalNextval);
				CHECK(equalChoose);
			}
			
		}

	}

}

#define FAST_SAMPLE(LAYER, COOR) LAYER->retrieve(COOR[0], COOR[1], COOR[2])

SCENARIO_METHOD(STPLayerManager, "STPLayerManager with some testing layers for biome generation", "[Diversity][STPLayerManager][!mayfail]") {

	GIVEN("A new layer manager") {

		THEN("The new layer manager should be empty") {
			REQUIRE(this->getLayerCount() == 0ull);
		}

		AND_GIVEN("Some random numbers as seeds") {
			using std::make_pair;
			//we are not using RNG, so doesn't matter
			constexpr Seed RandomSeed = 0ull;
			constexpr Seed Salt = 0ull;

			WHEN("Insert one layer with the unsupported cache size") {

				THEN("Layer manager does not allow such layer to be inserted") {
					REQUIRE_THROWS_AS((this->insert<RootLayer, 3ull>(RandomSeed, Salt)), STPException::STPBadNumericRange);
				}

			}

			WHEN("Insert one simple layer without cache") {
				auto FirstLayer = this->insert<RootLayer>(RandomSeed, Salt);
				const auto Coordinate = GENERATE(take(2, chunk(3, random(-13131313, 78987678))));

				THEN("The orphan layer properties should be validated") {
					//property test
					REQUIRE(this->getLayerCount() == 1ull);

					REQUIRE(FirstLayer->cacheSize() == 0ull);
					REQUIRE_FALSE(FirstLayer->hasAscendant());
					REQUIRE(FirstLayer->getAscendantCount() == 0ull);
					REQUIRE(FirstLayer->getAscendant() == nullptr);
					REQUIRE_FALSE(FirstLayer->isMerging());
				}

				THEN("The only layer should be the starting layer and can be sampled") {
					//pointer comparison to make sure they are the same layer from the same memory location
					REQUIRE(this->start() == FirstLayer);

					const auto LayerSample = FAST_SAMPLE(FirstLayer, Coordinate);

					AND_THEN("The sample should be deterministically random") {
						//same input always produces the same output
						REQUIRE(FAST_SAMPLE(FirstLayer, Coordinate) == LayerSample);
						//different input should produce different output (hash collision may fail it)
						CHECK_FALSE(FAST_SAMPLE(FirstLayer, -Coordinate) == LayerSample);
					}

				}

				AND_WHEN("Insert one more cached layer so the chain is linear") {
					auto SecondLayer = this->insert<NormalLayer, 32ull>(RandomSeed, Salt, FirstLayer);

					THEN("The properties of the new layer should be validated") {
						REQUIRE(this->getLayerCount() == 2ull);

						REQUIRE(SecondLayer->cacheSize() == 32ull);
						REQUIRE(SecondLayer->hasAscendant());
						REQUIRE(SecondLayer->getAscendantCount() == 1ull);
						REQUIRE(SecondLayer->getAscendant() == FirstLayer);
						REQUIRE_FALSE(SecondLayer->isMerging());
					}

					THEN("The starting layer should be the last layer inserted") {
						REQUIRE(this->start() == SecondLayer);

						const auto LayerSample = FAST_SAMPLE(SecondLayer, Coordinate);

						AND_THEN("The output sample value is correct") {
							//deterministic
							REQUIRE(FAST_SAMPLE(SecondLayer, Coordinate) == LayerSample);
							//random
							CHECK_FALSE(FAST_SAMPLE(SecondLayer, -Coordinate) == LayerSample);
							//our bottom layer simply call the sample function from the parent
							REQUIRE(LayerSample == FAST_SAMPLE(FirstLayer, Coordinate));
						}
					}

					AND_WHEN("Insert more layers to form a tree structure") {
						auto BranchLayer1 = this->insert<NormalLayer, 32ull>(RandomSeed, Salt, SecondLayer);
						auto BranchLayer2 = this->insert<NormalLayer>(RandomSeed, Salt, SecondLayer);
						auto MergeLayer = this->insert<MergingLayer, 32ull>(RandomSeed, Salt, BranchLayer1, BranchLayer2);

						THEN("The properties of all tree layers are validated") {
							REQUIRE(this->getLayerCount() == 5ull);

							REQUIRE(MergeLayer->isMerging());
							REQUIRE(MergeLayer->getAscendantCount() == 2ull);
							REQUIRE((MergeLayer->getAscendant(0) == BranchLayer1 && MergeLayer->getAscendant(1) == BranchLayer2));
						}

						THEN("Starting layer should be the last layer inserted, which is a merging layer") {
							REQUIRE(this->start() == MergeLayer);

							const auto BranchSample1 = FAST_SAMPLE(BranchLayer1, Coordinate);
							const auto BranchSample2 = FAST_SAMPLE(BranchLayer2, Coordinate);
							const auto TreeSample = FAST_SAMPLE(MergeLayer, Coordinate);

							AND_THEN("The output from the tree should be correct") {
								REQUIRE(FAST_SAMPLE(MergeLayer, Coordinate) == TreeSample);
								CHECK_FALSE(FAST_SAMPLE(MergeLayer, -Coordinate) == TreeSample);
								REQUIRE(TreeSample == static_cast<Sample>(BranchSample1 + BranchSample2));
							}
						}
					}
				}
			}

		}

	}

}

class BiomeFactoryTester : protected STPBiomeFactory {
protected:

	constexpr static uvec2 Dimension = uvec2(4u);
	constexpr static unsigned int PixelCount = Dimension.x * Dimension.y;
	constexpr static Seed RandomSeed = 0ull;
	constexpr static Seed Salt = 0ull;

	STPLayerManager supply() const override {
		STPLayerManager Mgr;
		STPLayer* Layer, *BranchLayer1, *BranchLayer2;
		
		Layer = Mgr.insert<RootLayer, 32ull>(BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt);
		Layer = Mgr.insert<NormalLayer>(BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);

		BranchLayer1 = Mgr.insert<NormalLayer, 32ull>(BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);
		BranchLayer2 = Mgr.insert<NormalLayer, 32ull>(BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);

		Layer = Mgr.insert<MergingLayer>(BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, BranchLayer1, BranchLayer2);

		return Mgr;
	}

	inline Sample getExpected(unsigned int index, const ivec3& offset) const {
		return static_cast<Sample>(((index % BiomeFactoryTester::Dimension.x) + (index / BiomeFactoryTester::Dimension.y) + offset.x + offset.z + 1) * 2);
	}

public:

	BiomeFactoryTester() : STPBiomeFactory(BiomeFactoryTester::Dimension) {

	}

	BiomeFactoryTester(uvec3 dimension) : STPBiomeFactory(dimension) {

	}

	void generateMap(Sample* biomemap, const ivec3& offset) {
		(*this)(biomemap, offset);
	}

};

SCENARIO_METHOD(BiomeFactoryTester, "STPBiomeFactory can be used to produce biomemap in batches", "[Diversity][STPBiomeFactory]") {

	WHEN("Dimension is some invalid values") {

		THEN("Biome factory should reject the values") {
			REQUIRE_THROWS_AS(BiomeFactoryTester(uvec3(0u, 4u, 8u)), SuperTerrainPlus::STPException::STPBadNumericRange);
		}

	}

	GIVEN("A complete biome factory with generation pipeline loaded") {

		THEN("Biome dimension should be available in the factory") {
			//2D biome generation should have the y dimension 1
			REQUIRE(this->BiomeDimension == uvec3(BiomeFactoryTester::Dimension.x, 1u, BiomeFactoryTester::Dimension.y));
		}

		AND_GIVEN("A world coordinate for generation") {
			using std::unique_ptr;
			using std::make_unique;
			unique_ptr<Sample[]> BiomeMap = make_unique<Sample[]>(BiomeFactoryTester::PixelCount);
			unique_ptr<Sample[]> AnotherBiomeMap = make_unique<Sample[]>(BiomeFactoryTester::PixelCount);

			WHEN("Asking the factory to generate a volumetric biomemap") {

				THEN("Biome factory current does not support such texture") {
					BiomeFactoryTester BrokenFactory(uvec3(2u, 128u, 2u));
					REQUIRE_THROWS_AS(BrokenFactory.generateMap(BiomeMap.get(), ivec3(-564738, 476329, -659843)), STPException::STPUnsupportedFunctionality);
				}

			}

			const auto Coordinate = GENERATE(take(1, chunk(2, random(-666666666, 666666666))));
			const ivec3 Offset = ivec3(Coordinate[0], GENERATE(values({ 0, 16974382 })), Coordinate[1]);
			WHEN("Asking the factory to generate a new " << (Offset.y == 0 ? "2" : "3") << "D flat biomemap") {

				THEN("Generation should be successful") {
					REQUIRE_NOTHROW((*this)(BiomeMap.get(), Offset));
					//because our biomemap has memory pool, just to test if the memory has been reused (coverage test will show it)
					REQUIRE_NOTHROW((*this)(AnotherBiomeMap.get(), Offset));

					AND_THEN("The biomemap should be validated") {
						//root layer is ((x_offset + y_offset) + x + y + z + 1), normal layer simply uses the value from parent, merging layer adds the values
						//and in 2D biome generator y is ignore (essentially treated as 0)
						//3D biome, currently behaves the same as 2D biome because it's not yet implemented
						const unsigned int Index = GENERATE(take(3, random(0u, 15u)));
						REQUIRE(BiomeMap[Index] == this->getExpected(Index, Offset));
						REQUIRE(AnotherBiomeMap[Index] == BiomeMap[Index]);
					}
				}

			}

		}

	}

}