#include "catch2/catch.hpp"

#include "../World/Biome/STPSeedMixer.cpp"
#include "../World/Biome/STPLayerCache.cpp"
#include "../World/Biome/STPLayer.cpp"

using namespace Catch::Generators;
using Catch::Matchers::EndsWith;
using Catch::Matchers::Contains;
using namespace SuperTerrainPlus::STPBiome;

class arbitary_biome : public STPLayer {
public:

	arbitary_biome(Seed global, Seed salt) : STPLayer(global, salt) {

	}

	Sample sample(int x, int y, int z) override {
		return static_cast<Sample>(x + y + z);
	}

};

class localseed_biome : public STPLayer {
public:

	localseed_biome(Seed global, Seed salt) : STPLayer(global, salt) {

	}

	Sample sample(int x, int y, int z) override {
		return static_cast<Sample>(this->genLocalSeed(x, z) & 0xFFFEu);
	}

};

class rng_biome : public STPLayer {
public:

	rng_biome(Seed global, Seed salt) : STPLayer(global, salt) {

	}

	Sample sample(int x, int y, int z) override {
		const STPLayer::STPLocalRNG rng = this->getRNG(this->genLocalSeed(x, z));
		return rng.nextVal(0xFFFFu);
	}

};

struct layer_test {
private:

	STPLayer* b = nullptr;

protected:

	Sample sample(int x, int y, int z) {
		return this->b->sample(x, y, z);
	}

	Sample sample_cached(int x, int y, int z) {
		return this->b->sample_cached(x, y, z);
	}

	size_t asc_size() {
		return this->b->getAscendantCount();
	}

public:

	layer_test() {
		this->b = STPLayer::create<arbitary_biome, 512u>(random(0u, 100000u).get(), random(0u, 100000u).get());
	}

	~layer_test() {
		STPLayer::destroy(this->b);
	}

};

class layer_node : public STPLayer {
private:

	static Seed gen() {
		return random(0u, 100000u).get();
	}

public:

	layer_node(Seed global, Seed salt) : STPLayer(global, salt) {

	}

	layer_node(Seed global, Seed salt, STPLayer* asc) : STPLayer(global, salt, asc) {

	}
	
	layer_node(Seed global, Seed salt, STPLayer* asc1, STPLayer* asc2) : STPLayer(global, salt, asc1, asc2) {

	}

	Sample sample(int x, int y, int z) {
		if (this->hasAscendant()) {
			if (this->getAscendantCount() > 1) {
				return static_cast<Sample>(x + y + z) + this->getAscendant(0)->sample(x, y, z) + this->getAscendant(1)->sample(x, y, z);
			}
			return static_cast<Sample>(x + y + z) + this->getAscendant()->sample(x, y, z);
		}
		return static_cast<Sample>(x + y + z);
	}

	static STPLayer* direct_tree() {
		STPLayer* tree = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen());
		tree = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree);
		tree = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree);

		return tree;
	}

	static STPLayer* merging_only() {
		STPLayer* tree1 = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen());
		STPLayer* tree2 = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen());

		STPLayer* tree = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree1, tree2);

		return tree;
	}

	static STPLayer* merging_branching() {
		STPLayer* tree_top = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen());

		STPLayer* tree1 = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree_top);
		STPLayer* tree2 = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree_top);

		STPLayer* tree = STPLayer::create<layer_node>(layer_node::gen(), layer_node::gen(), tree1, tree2);

		return tree;
	}
};

TEST_CASE_METHOD(layer_test, "Biome basic framework test", "[STPBiome]") {

	SECTION("Seed mixer functionality test") {
		const Seed input = GENERATE(take(5, random(0u, 100000u)));
		const Seed output = STPSeedMixer::mixSeed(input, GENERATE(take(5, random(0u, 100000u))));
		
		REQUIRE(input != output);
	}

	SECTION("----- Layer cache tests -----") {
		
		SECTION("Layer and cache creation and destroy") {
			REQUIRE_NOTHROW([]() -> void {
				STPLayer* layer = STPLayer::create<arbitary_biome, 1024ull>(random(0u, 100000u).get(), random(0u, 100000u).get());
				STPLayer::destroy(layer);
				return;
				}());
		}

		SECTION("Layer creation with illegeal non-zero cache size") {
			REQUIRE_THROWS_WITH((STPLayer::create<arbitary_biome, 100ull>(random(0u, 100000u).get(), random(0u, 100000u).get())), Contains("power of 2"));
		}

		SECTION("Layer creation when cache size is zero") {
			REQUIRE_NOTHROW([]() -> void {
				STPLayer* layer = STPLayer::create<arbitary_biome, 0ull>(random(0u, 100000u).get(), random(0u, 100000u).get());
				STPLayer::destroy(layer);
				return;
				}());
		}

		SECTION("Simple read from cache with cache miss") {
			const int x = GENERATE(take(5, random(0, 1000)));
			const int y = GENERATE(take(5, random(0, 1000)));
			const int z = GENERATE(take(5, random(0, 1000)));

			REQUIRE(sample_cached(x, y, z) == x + y + z);
		}
		
		SECTION("Simple read from cache with cache hit") {
			const int x = random(-10000, 10000).get();
			const int y = random(-10000, 10000).get();
			const int z = random(-10000, 10000).get();

			for (int i = 0; i < 5; i++) {
				REQUIRE(sample_cached(x, y, z) == x + y + z);
			}
		}
		
	}

	SECTION("----- Layer basic functional test -----") {

		SECTION("Correct output of number of ascendant") {
			REQUIRE(asc_size() == 0ull);
		}

		SECTION("Simple sampling method override") {
			const int x = GENERATE(take(5, random(0, 1000)));
			const int y = GENERATE(take(5, random(0, 1000)));
			const int z = GENERATE(take(5, random(0, 1000)));

			REQUIRE(sample(x, y, z) == x + y + z);
		}

		SECTION("Randomness of layer seed generation") {
			Seed prev;
			for (int i = 0; i < 6; i++) {
				const Seed global = random(0u, 100000u).get();
				const Seed salt = random(0u, 100000u).get();
				STPLayer* layer = STPLayer::create<arbitary_biome, 0ull>(global, salt);

				if (i != 0) {
					INFO("Layer seed should not be equal, got " << prev << " == " << layer->LayerSeed);
					REQUIRE_FALSE(prev == layer->LayerSeed);
				}
				prev = layer->LayerSeed;

				STPLayer::destroy(layer);
			}
		}

		SECTION("Deterministisity of layer seed generation") {
			for (int i = 0; i < 5; i++) {
				const Seed global = random(0u, 100000u).get();
				const Seed salt = random(0u, 100000u).get();
				STPLayer* layer1 = STPLayer::create<arbitary_biome, 0ull>(global, salt);
				STPLayer* layer2 = STPLayer::create<arbitary_biome, 0ull>(global, salt);

				INFO("Layer seed should be equal when seeds are the same: got " << layer1->LayerSeed << " and " << layer2->LayerSeed);
				REQUIRE(layer1->LayerSeed == layer2->LayerSeed);

				STPLayer::destroy(layer1);
				STPLayer::destroy(layer2);
			}
		}

		SECTION("Randomness of local seed generation") {
			Sample prev;
			const Seed global = random(0u, 100000u).get();
			const Seed salt = random(0u, 100000u).get();
			for (int i = 0; i < 6; i++) {
				const int x = random(-10000, 10000).get();
				const int z = random(-10000, 10000).get();
				STPLayer* layer = STPLayer::create<localseed_biome, 0ull>(global, salt);

				const Sample sample = layer->sample(x, 0, z);
				if (i != 0) {
					INFO("Local seed should not be equal, got " << prev << " == " << sample);
					REQUIRE_FALSE(prev == sample);
				}
				prev = sample;

				STPLayer::destroy(layer);
			}
		}

		SECTION("Deterministisity of local seed generation") {
			for (int i = 0; i < 5; i++) {
				const Seed global = random(0u, 100000u).get();
				const Seed salt = random(0u, 100000u).get();
				const int x = random(-10000, 10000).get();
				const int z = random(-10000, 10000).get();
				STPLayer* layer1 = STPLayer::create<localseed_biome, 0ull>(global, salt);
				STPLayer* layer2 = STPLayer::create<localseed_biome, 0ull>(global, salt);

				const Sample seed1 = layer1->sample(x, 0, z);
				const Sample seed2 = layer2->sample(x, 0, z);
				INFO("Local seed should be equal when inputs are the same: got " << seed1 << " and " << seed2);
				REQUIRE(seed1 == seed2);

				STPLayer::destroy(layer1);
				STPLayer::destroy(layer2);
			}
		}

		SECTION("Randomness of local random number generator") {
			Sample prev;
			const Seed global = random(0u, 100000u).get();
			const Seed salt = random(0u, 100000u).get();
			for (int i = 0; i < 6; i++) {
				const int x = random(-10000, 10000).get();
				const int z = random(-10000, 10000).get();
				STPLayer* layer = STPLayer::create<rng_biome, 0ull>(global, salt);

				const Sample sample = layer->sample(x, 0, z);
				if (i != 0) {
					INFO("Rng result should not be equal, got " << prev << " == " << sample);
					REQUIRE_FALSE(prev == sample);
				}
				prev = sample;

				STPLayer::destroy(layer);
			}
		}

		SECTION("Deterministisity of local random number generator") {
			for (int i = 0; i < 5; i++) {
				const Seed global = random(0u, 100000u).get();
				const Seed salt = random(0u, 100000u).get();
				const int x = random(-10000, 10000).get();
				const int z = random(-10000, 10000).get();
				STPLayer* layer1 = STPLayer::create<localseed_biome, 0ull>(global, salt);
				STPLayer* layer2 = STPLayer::create<localseed_biome, 0ull>(global, salt);

				const Sample seed1 = layer1->sample(x, 0, z);
				const Sample seed2 = layer2->sample(x, 0, z);
				INFO("Rng result should be equal when inputs are the same: got " << seed1 << " and " << seed2);
				REQUIRE(seed1 == seed2);

				STPLayer::destroy(layer1);
				STPLayer::destroy(layer2);
			}
		}
	}

	SECTION("----- Layer tree structure test -----") {

		SECTION("Simple tree construction and destruction test") {
			REQUIRE_NOTHROW([]() -> void {
				STPLayer* tree = layer_node::direct_tree();
				STPLayer::destroy(tree);
				return;
				}());
		}

		SECTION("Correct output of number of ascendant") {
			STPLayer* tree = layer_node::direct_tree();
			STPLayer* next = tree;

			REQUIRE(next->getAscendantCount() == 1);
			next = next->getAscendant();
			REQUIRE(next->getAscendantCount() == 1);
			next = next->getAscendant();
			REQUIRE(next->getAscendantCount() == 0);
			REQUIRE(next->getAscendant() == nullptr);
			
			STPLayer::destroy(tree);
		}


		SECTION("Correctly determine merging layer") {
			STPLayer* tree = layer_node::merging_only();

			REQUIRE(tree->isMerging());
			REQUIRE_FALSE(tree->getAscendant(0)->isMerging());

			STPLayer::destroy(tree);
		}

		SECTION("Tree with only one straight branch") {
			const int x = GENERATE(take(5, random(0, 1000)));
			const int y = GENERATE(take(5, random(0, 1000)));
			const int z = GENERATE(take(5, random(0, 1000)));

			STPLayer* tree = layer_node::direct_tree();

			const Sample sample = tree->sample(x, y, z);
			REQUIRE(sample == (x + y + z) * 3);

			STPLayer::destroy(tree);
		}

		SECTION("Tree with two branches and they merge to a single node") {
			const int x = GENERATE(take(5, random(0, 1000)));
			const int y = GENERATE(take(5, random(0, 1000)));
			const int z = GENERATE(take(5, random(0, 1000)));

			STPLayer* tree = layer_node::merging_only();

			const Sample sample = tree->sample(x, y, z);
			REQUIRE(sample == (x + y + z) * 3);

			STPLayer::destroy(tree);
		}

		SECTION("Tree that diverges and converges, a.k.a., branch then merge") {
			const int x = GENERATE(take(5, random(0, 1000)));
			const int y = GENERATE(take(5, random(0, 1000)));
			const int z = GENERATE(take(5, random(0, 1000)));

			STPLayer* tree = layer_node::merging_branching();

			const Sample sample = tree->sample(x, y, z);
			REQUIRE(sample == (x + y + z) * 5);

			STPLayer::destroy(tree);
		}
	}
}