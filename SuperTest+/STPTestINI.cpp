//Catch2
#include <catch2/catch_test_macros.hpp>
//Generator
#include <catch2/generators/catch_generators_range.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//SuperAlgorithm+/Parser/INI
#include <SuperAlgorithm+/Parser/INI/STPINIBasicString.h>
#include <SuperAlgorithm+/Parser/INI/STPINIReader.h>
#include <SuperAlgorithm+/Parser/INI/STPINIWriter.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

#include <optional>
#include <utility>

using std::optional;
using std::string;
using std::string_view;
using std::pair;
using std::make_pair;

constexpr static char TestINIFilename[] = "./TestData/Data.ini";

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPAlgorithm;

using namespace Catch::Matchers;

SCENARIO("INI string utility can convert string to primitive value", "[AlgorithmHost][INI][STPINIBasicString]") {

	GIVEN("A string containing valid primitive data") {

		WHEN("User wants to convert it to a corresponding value") {

			THEN("The conversion produces correct result") {
				CHECK(STPINIStringView("-13").to<int>() == -13);
				CHECK(STPINIStringView("1313l").to<long>() == 1313l);
				CHECK(STPINIStringView("666666ll").to<long long>() == 666666ll);
				CHECK(STPINIStringView("54u").to<unsigned int>() == 54u);
				CHECK(STPINIStringView("8888ul").to<unsigned long>() == 8888ul);
				CHECK(STPINIStringView("123456789ull").to<unsigned long long>() == 123456789ull);

				CHECK_THAT(STPINIStringView("2.718f").to<float>(), WithinAbs(2.718f, 1e-4f));
				CHECK_THAT(STPINIStringView("1.414213562").to<double>(), WithinAbs(1.414213562, 1e-10));
			}

		}

	}

}

SCENARIO("INI reader can parsed all values correctly", "[AlgorithmHost][INI][STPINIReader]") {
	optional<STPINIReader> Reader;

	GIVEN("A syntactically correct raw INI string") {
		const string TestINIString = STPFile::read(TestINIFilename);

		WHEN("It is fed into the INI reader") {

			THEN("It can be parsed successfully without errors") {
				REQUIRE_NOTHROW([&Reader, &TestINIString]() { Reader.emplace(TestINIString); }());

				AND_THEN("All parsed values are correct") {
					const auto& Storage = **Reader;
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
				}

			}

		}

	}

	GIVEN("A raw INI string which does not comply with INI syntax") {
		auto tryRead = [&Reader](const char* src) {
			Reader.emplace(src);
		};
		constexpr char BrokenINI1[] = "[Diet\ncat=fish";
		constexpr char BrokenINI2[] = "[Diet]\ncat fish";

		WHEN("It is fed into the INI parser") {

			THEN("It reports error") {
				CHECK_THROWS_WITH(tryRead(BrokenINI1), ContainsSubstring("Diet"));
				CHECK_THROWS_WITH(tryRead(BrokenINI2), ContainsSubstring("cat fish"));
			}

		}

	}

}

SCENARIO("INI writer can format all data as an INI document", "[AlgorithmHost][INI][STPINIWriter]") {
	
	GIVEN("A data structure of INI") {
		//As long as all string data are literals, it is fine to use view
		STPINIStorageView Storage;
		STPINISectionView& NamelessSec = Storage[""];
		STPINISectionView& FoodSec = Storage["Food"];
		STPINISectionView& DrinkSec = Storage["Drink"];

		NamelessSec["location"] = STPINIStringView("SuperRestaurant+");
		NamelessSec["invitee"] = STPINIStringView("123u");

		FoodSec["appetiser"] = STPINIStringView("chicken salad");
		FoodSec["soup"] = STPINIStringView("mushroom");
		FoodSec["main course"] = STPINIStringView("beef fillet");
		FoodSec["dessert"] = STPINIStringView("ice-cream");

		DrinkSec["soda"] = STPINIStringView("orange taste");
		DrinkSec["water"] = STPINIStringView("sparkling");

		WHEN("The data structure is fed into an INI writer") {
			optional<STPINIWriter> Writer;

			THEN("The writer outputs the INI raw string") {
				REQUIRE_NOTHROW([&Writer, &Storage]() { Writer.emplace(Storage); }());
				const string& Output = **Writer;

				CHECK_THAT(Output, ContainsSubstring("[Food]"));
				CHECK_THAT(Output, ContainsSubstring("[Drink]"));

				CHECK_THAT(Output, ContainsSubstring("location=SuperRestaurant+"));
				CHECK_THAT(Output, ContainsSubstring("invitee=123u"));

				CHECK_THAT(Output, ContainsSubstring("appetiser=chicken salad"));
				CHECK_THAT(Output, ContainsSubstring("soup=mushroom"));
				CHECK_THAT(Output, ContainsSubstring("main course=beef fillet"));
				CHECK_THAT(Output, ContainsSubstring("dessert=ice-cream"));

				CHECK_THAT(Output, ContainsSubstring("soda=orange taste"));
				CHECK_THAT(Output, ContainsSubstring("water=sparkling"));
			}

			AND_WHEN("The writer is also told to format the output") {
				const auto Trial = GENERATE(range(0, 3));
				constexpr STPINIWriter::STPWriterFlag AllFlag[] = {
					STPINIWriter::SpaceAroundAssignment,
					STPINIWriter::SpaceAroundSectionName | STPINIWriter::SectionNewline,
					STPINIWriter::SpaceAroundAssignment | STPINIWriter::SpaceAroundSectionName
				};
				constexpr pair<string_view, string_view> AllOutput[] = {
					make_pair("\n[Food]", "main course = beef fillet"),
					make_pair("\n\n[ Food ]", "main course=beef fillet"),
					make_pair("\n[ Food ]", "main course = beef fillet"),
				};

				REQUIRE_NOTHROW([&Writer, &Storage, ChosenFlag = AllFlag[Trial]]() { Writer.emplace(Storage, ChosenFlag); }());
				const string& Output = **Writer;

				THEN("The writer also has the ability to format the output") {
					const auto& [Expected1, Expected2] = AllOutput[Trial];

					CHECK_THAT(Output, ContainsSubstring(string(Expected1)));
					CHECK_THAT(Output, ContainsSubstring(string(Expected2)));
				}
			}

		}

	}

}