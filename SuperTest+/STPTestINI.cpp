//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators_range.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//SuperAlgorithm+/Parser/INI
#include <SuperAlgorithm+/Parser/STPINIParser.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

using std::string_view;
using std::string;

constexpr static string_view TestINIFilename = "./TestData/Data.ini";

namespace STPFile = SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPAlgorithm;

using namespace Catch::Matchers;

SCENARIO("INI reader can parsed all values correctly", "[AlgorithmHost][INI][STPINIReader]") {
	STPINIParser::STPINIReaderResult Result;
	const auto& [Storage, SecOrder, PropOrder] = Result;

	GIVEN("A syntactically correct raw INI string") {
		const string TestINIString = STPFile::read(TestINIFilename.data());

		WHEN("It is fed into the INI reader") {

			THEN("It can be parsed successfully without errors") {
				REQUIRE_NOTHROW([&Result, &TestINIString]() {
					Result = STPINIParser::read(TestINIString, TestINIFilename);
				}());

				AND_THEN("All parsed values are correct") {
					const auto& NamelessSec = Storage.at("");
					const auto& DietSec = Storage.at("Diet");
					const auto& MakeSoundSec = Storage.at("Make Sound");

					REQUIRE(Storage.size() == 3u);
					REQUIRE(NamelessSec.size() == 3u);
					REQUIRE(DietSec.size() == 4u);
					REQUIRE(MakeSoundSec.size() == 6u);

					CHECK((NamelessSec.at("day").String == "Monday"));
					CHECK((NamelessSec.at("weather").String == "sunny"));
					CHECK((NamelessSec.at("temperature").String == "16.5f"));

					CHECK((DietSec.at("cat").String == "fish"));
					CHECK((DietSec.at("dog").String == "tasty bone"));
					CHECK((DietSec.at("cow").String == "grass"));
					CHECK((DietSec.at("pig").String == "omnivore"));

					CHECK((MakeSoundSec.at("cat").String == "meow"));
					CHECK((MakeSoundSec.at("dog").String == "bark"));
					CHECK((MakeSoundSec.at("cow").String == "moo"));
					CHECK((MakeSoundSec.at("pig").String == "oink"));
					CHECK((MakeSoundSec.at("tiger").String == "roar"));
					CHECK((MakeSoundSec.at("birds").String == "sing"));

					//order 
					CHECK(SecOrder.at("Diet") == 1u);
					CHECK(SecOrder.at("Make Sound") == 2u);

					CHECK(PropOrder.at(0).at("day") == 0u);
					CHECK(PropOrder.at(1).at("cow") == 2u);
					CHECK(PropOrder.at(2).at("birds") == 5u);
				}

			}

		}

	}

	GIVEN("A raw INI string which does not comply with INI syntax") {
		const auto tryRead = [&Result](const char* const src) {
			constexpr static string_view BrokenINIFilename = "BrokenData.ini";
			Result = STPINIParser::read(src, BrokenINIFilename);
		};
		constexpr char BrokenINI1[] = "[Diet\ncat=fish";
		constexpr char BrokenINI2[] = "[Diet]\ncat fish";

		WHEN("It is fed into the INI parser") {

			THEN("It reports error") {
				CHECK_THROWS_WITH(tryRead(BrokenINI1), ContainsSubstring("SectionLineClosing"));
				CHECK_THROWS_WITH(tryRead(BrokenINI2), ContainsSubstring("KeyValueSeparator"));
			}

		}

	}

}