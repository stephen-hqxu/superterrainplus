//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

//SuperTerrain+/SuperTerrain+/World/Diversity
#include<SuperTerrain+/World/Diversity/STPLayer.h>
#include <SuperTerrain+/World/Diversity/STPLayerTree.h>
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

class RootLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		//unsigned type wins!!!
		return static_cast<Sample>(1u + x + y + z);
	}

public:

	RootLayer(size_t cache, Seed seed, Seed salt) : STPLayer(cache, seed, salt) {

	}

};

class NormalLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		//simply use the value from the parent
		return this->getAscendant()->retrieve(x, y, z);
	}

public:

	NormalLayer(size_t cache, Seed seed, Seed salt, STPLayer* ascendant) : STPLayer(cache, seed, salt, ascendant) {

	}

};

class MergingLayer : public STPLayer {
protected:

	Sample sample(int x, int y, int z) override {
		return static_cast<Sample>(this->getAscendant(0)->retrieve(x, y, z) + this->getAscendant(1)->retrieve(x, y, z));
	}

public:

	MergingLayer(size_t cache, Seed seed, Seed salt, STPLayer* asc1, STPLayer* asc2) : STPLayer(cache, seed, salt, asc1, asc2) {

	}

};

class RandomLayer : public STPLayer {
private:

	Sample sample(int x, int, int z) {
		return static_cast<Sample>(this->seedLocal(x, z));
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

	RandomLayer() : STPLayer(0u, 19657483ull, 48321567ull) {
		
	}

};

SCENARIO_METHOD(RandomLayer, "STPLayer generates random seed with built-in RNG", "[Diversity][STPLayer][!mayfail]") {

	GIVEN("A random seed to be mixed with another seed") {
		const auto SeedSalt = GENERATE(take(3, chunk(2, random(-123456789876ll, 66666666666ll))));
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
			const Seed LocalSeed = this->seedLocal(Coord.x, Coord.y);

			THEN("The RNG loaded with the same local seed should produce the same sequence of output") {
				constexpr static unsigned int BucketSize = 16u;
				constexpr static Sample Bound = std::numeric_limits<Sample>::max();

				const STPLayer::STPLocalSampler RNG1 = this->createLocalSampler(LocalSeed),
					RNG2 = this->createLocalSampler(LocalSeed);

				bool equalNextval = true, equalChoose = true;
				for (unsigned int i = 0u; i < BucketSize && (equalNextval || equalChoose); i++) {
					equalNextval &= RNG1.nextValue(Bound) == RNG2.nextValue(Bound);
					equalChoose &= RNG1.choose(0u, 1u, 2u, 3u) == RNG2.choose(0u, 1u, 2u, 3u);
				}
				CHECK(equalNextval);
				CHECK(equalChoose);
			}
			
		}

	}

}

#define FAST_SAMPLE(LAYER, COOR) LAYER->retrieve(COOR[0], COOR[1], COOR[2])

SCENARIO_METHOD(STPLayerTree, "STPLayerTree with some testing layers for biome generation", "[Diversity][STPLayerTree][!mayfail]") {

	GIVEN("A new layer manager") {

		THEN("The new layer manager should be empty") {
			REQUIRE(this->getLayerCount() == 0u);
		}

		AND_GIVEN("Some random numbers as seeds") {
			using std::make_pair;
			//we are not using RNG, so doesn't matter
			constexpr Seed RandomSeed = 0ull;
			constexpr Seed Salt = 0ull;

			WHEN("Insert one layer with the unsupported cache size") {

				THEN("Layer manager does not allow such layer to be inserted") {
					REQUIRE_THROWS_AS((this->insert<RootLayer>(3u, RandomSeed, Salt)), STPException::STPBadNumericRange);
				}

			}

			WHEN("Insert one simple layer without cache") {
				auto FirstLayer = this->insert<RootLayer>(0u, RandomSeed, Salt);
				const auto Coordinate = GENERATE(take(2, chunk(3, random(-13131313, 78987678))));

				THEN("The orphan layer properties should be validated") {
					//property test
					REQUIRE(this->getLayerCount() == 1u);

					REQUIRE(FirstLayer->cacheSize() == 0u);
					REQUIRE(FirstLayer->AscendantCount == 0u);
					REQUIRE(FirstLayer->getAscendant() == nullptr);
					REQUIRE(FirstLayer->getAscendant(123u) == nullptr);
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
					auto SecondLayer = this->insert<NormalLayer>(32u, RandomSeed, Salt, FirstLayer);

					THEN("The properties of the new layer should be validated") {
						REQUIRE(this->getLayerCount() == 2u);

						REQUIRE(SecondLayer->cacheSize() == 32u);
						REQUIRE(SecondLayer->AscendantCount == 1u);
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
						auto BranchLayer1 = this->insert<NormalLayer>(32u, RandomSeed, Salt, SecondLayer);
						auto BranchLayer2 = this->insert<NormalLayer>(0u, RandomSeed, Salt, SecondLayer);
						auto MergeLayer = this->insert<MergingLayer>(32u, RandomSeed, Salt, BranchLayer1, BranchLayer2);

						THEN("The properties of all tree layers are validated") {
							REQUIRE(this->getLayerCount() == 5u);

							REQUIRE(MergeLayer->isMerging());
							REQUIRE(MergeLayer->AscendantCount == 2u);
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

	STPLayerTree supply() const override {
		STPLayerTree Tree;
		STPLayer* Layer, *BranchLayer1, *BranchLayer2;
		
		Layer = Tree.insert<RootLayer>(32u, BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt);
		Layer = Tree.insert<NormalLayer>(0u, BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);

		BranchLayer1 = Tree.insert<NormalLayer>(32u, BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);
		BranchLayer2 = Tree.insert<NormalLayer>(32u, BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, Layer);

		Layer = Tree.insert<MergingLayer>(0u, BiomeFactoryTester::RandomSeed, BiomeFactoryTester::Salt, BranchLayer1, BranchLayer2);

		return Tree;
	}

	inline Sample getExpected(unsigned int index, const ivec2& offset) const {
		return static_cast<Sample>(((index % BiomeFactoryTester::Dimension.x) + (index / BiomeFactoryTester::Dimension.y) + offset.x + offset.y + 1) * 2);
	}

public:

	BiomeFactoryTester() : STPBiomeFactory(BiomeFactoryTester::Dimension) {

	}

	BiomeFactoryTester(uvec2 dimension) : STPBiomeFactory(dimension) {

	}

	void generateMap(Sample* biomemap, const ivec2& offset) {
		(*this)(biomemap, offset);
	}

};

SCENARIO_METHOD(BiomeFactoryTester, "STPBiomeFactory can be used to produce biomemap in batches", "[Diversity][STPBiomeFactory]") {

	WHEN("Dimension is some invalid values") {

		THEN("Biome factory should reject the values") {
			REQUIRE_THROWS_AS(BiomeFactoryTester(uvec2(0u, 4u)), SuperTerrainPlus::STPException::STPBadNumericRange);
		}

	}

	GIVEN("A complete biome factory with generation pipeline loaded") {

		THEN("Biome dimension should be available in the factory") {
			REQUIRE(this->BiomeDimension == BiomeFactoryTester::Dimension);
		}

		AND_GIVEN("A world coordinate for generation") {
			Sample BiomeMap[BiomeFactoryTester::PixelCount];
			Sample AnotherBiomeMap[BiomeFactoryTester::PixelCount];

			const auto Coordinate = GENERATE(take(1, chunk(2, random(-666666666, 666666666))));
			const ivec2 Offset = ivec2(Coordinate[0], Coordinate[1]);
			WHEN("Asking the factory to generate a new 2D flat biomemap") {

				THEN("Generation should be successful") {
					REQUIRE_NOTHROW((*this)(BiomeMap, Offset));
					//because our biomemap has memory pool, just to test if the memory has been reused (coverage test will show it)
					REQUIRE_NOTHROW((*this)(AnotherBiomeMap, Offset));

					AND_THEN("The biomemap should be validated") {
						//root layer is ((x_offset + y_offset) + x + y + z + 1), normal layer simply uses the value from parent, merging layer adds the values
						//and in 2D biome generator y is ignore (essentially treated as 0)
						const unsigned int Index = GENERATE(take(3, random(0u, 15u)));
						REQUIRE(BiomeMap[Index] == this->getExpected(Index, Offset));
						REQUIRE(AnotherBiomeMap[Index] == BiomeMap[Index]);
					}
				}

			}

		}

	}

}