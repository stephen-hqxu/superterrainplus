//Catch2
#include <catch2/catch_test_macros.hpp>

//SuperAlgorithm+Host/Parser
#include <SuperAlgorithm+/Parser/Framework/STPRegularLanguage.h>

//String
#include <string_view>

using namespace SuperTerrainPlus::STPAlgorithm;

using std::string_view;
using std::nullopt;

namespace {
	namespace RL = STPRegularLanguage;
	namespace CC = RL::STPCharacterClass;
	namespace Q = RL::STPQuantifier;

	namespace SimpleMatcher {

		constexpr static string_view StringA = "a", StringB = "b";

		//atomic operator
		using MatchAny = RL::Any;
		using MatchLiteralA = RL::Literal<StringA>;
		using MatchLiteralB = RL::Literal<StringB>;
		//character class operator
		using MatchAorB = CC::Class<CC::Atomic<'a'>, CC::Atomic<'b'>>;
		using MatchUpper = CC::Class<CC::Range<'A', 'Z'>>;
		using MatchSymbol = CC::Class<CC::Except<CC::Range<'0', '9'>, CC::Range<'a', 'z'>, CC::Range<'A', 'Z'>>>;
		//repeat operator
		using MatchMaybeA = Q::Repeat<MatchLiteralA, 0u, 1u>;
		using MatchAtLeast2A = Q::Repeat<MatchLiteralA, 2u, Q::Unlimited>;
		using Match5A = Q::Repeat<MatchLiteralA, 5u>;
		//alternative operator
		using MatchAorBAlt = RL::Alternative<MatchLiteralA, MatchLiteralB>;
		//sequence operator
		using MatchABAny = RL::Sequence<MatchLiteralA, MatchLiteralB, MatchAny>;

	}

	namespace IPv6Matcher {

		//they are randomly generated, does not represent any IP address in real life
		constexpr static string_view Example[] = {
			"6d24:628c:82fd:334b:c6ac:5e7b:03ab:2cb5",//full
			"46b4:1032:77:48e8:671d:7f27:a04:a96a",//omitting 0
			"1234:5678:9012:3456:7890:1234:5678:9012",//all digits
			"abcd:efab:cdef:abcd:efab:cdef:abcd:efab",//all alphabets
			"127.0.0.1",//ipv4
			"e9be:69dd:2495",//not enough groups
			"0950:x123",//not a hex
			"1faf:5140:1382c",//too many digits in a group
			"435d:b52e:66fe:e103:46ee:d53f:1300:5bd3:739a",//too many groups
			"8d13:f8eb:1b3e:a973:4d2d:b216-decf:6a52",//wrong group separator
			"hello world"
		};
		constexpr static RL::STPMatchLength MatchLengthReference[] = {
			Example[0].length(), Example[1].length(), Example[2].length(), Example[3].length(),
			nullopt, nullopt, nullopt, nullopt, Example[8].length() - 5u, nullopt, nullopt
		};
		constexpr static bool ResultReference[] = {
			true, true, true, true, false, false, false, false, false, false, false
		};
		//size of bool is not required by the standard to be 1 byte
		constexpr static size_t ExampleCount = sizeof(ResultReference) / sizeof(bool);

		constexpr static string_view GroupSeparator = ":";

		//match like "01ad", 0 can be omitted
		using FourHexDigit =
			Q::Repeat<
				CC::Class<
					CC::Range<'0', '9'>,
					CC::Range<'a', 'f'>
				>,
				1u, 4u
			>;
		//match like ":01ad", does not allow omitting empty group
		using MiddleGroup =
			Q::Repeat<
				RL::Sequence<
					RL::Literal<GroupSeparator>,
					FourHexDigit
				>,
				7u
			>;
		using Main =
			RL::Sequence<
				FourHexDigit,
				MiddleGroup
			>;
	}

	namespace VariableDefinitionMatcher {

		constexpr static string_view Example[] = {
			"int pi = 31415926",//int and short are 2 valid types
			"short e = 271828",
			"const int data = 1234567890",//*const* is an optional qualifier
			"const short number = 6",
			"float = 0",//*float* is not a valid type
			"volatile int a = 0",//*volatile* is not a valid qualifier
			"int a_a = 0",//variable name should be alphabets
			"short a = a",//value should be numbers
			"short a : 0",//*:* is not defined
			"short a =0"//no space
		};
		constexpr static RL::STPMatchLength MatchLengthReference[] = {
			Example[0].length(), Example[1].length(), Example[2].length(), Example[3].length(),
			nullopt, nullopt, nullopt, nullopt, nullopt, nullopt
		};
		constexpr static bool ResultReference[] = {
			true, true, true, true, false, false, false, false, false, false
		};
		constexpr static size_t ExampleCount = sizeof(ResultReference) / sizeof(bool);

		constexpr static string_view QualConst = "const", TypeInt = "int", TypeShort = "short",
			Space = " ", Equal = "=";

		using SpaceLiteral =
			RL::Literal<Space>;
		using MaybeConst =
			Q::Maybe<
				RL::Sequence<
					RL::Literal<QualConst>,
					SpaceLiteral
				>
			>;
		using Type =
			RL::Alternative<
				RL::Literal<TypeInt>,
				RL::Literal<TypeShort>
			>;
		using Main =
			RL::Sequence<
				MaybeConst,
				Type,
				SpaceLiteral,
				Q::StrictMany<
					CC::Class<
						CC::Range<'a', 'z'>
					>
				>,//variable name
				SpaceLiteral,
				RL::Literal<Equal>,
				SpaceLiteral,
				Q::StrictMany<
					CC::Class<
						CC::Range<'0', '9'>
					>
				>//value
			>;
	}
}

//we consider a match if the input and output are exactly the same in length
inline constexpr static bool isValidMatch(const string_view& input, const RL::STPMatchLength& match_length) noexcept {
	return match_length && *match_length == input.length();
}

#define RUN_MATCHER_TEST do { \
	for (size_t i = 0u; i < ExampleCount; i++) { \
		const string_view& example = Example[i]; \
		const RL::STPMatchLength match_length = Main::match(example); \
		REQUIRE(isValidMatch(example, match_length) == ResultReference[i]); \
		CHECK(match_length == MatchLengthReference[i]); \
	} \
} while (false)

SCENARIO("STPRegularLanguage can match string based on defined matching expressions", "[AlgorithmHost][STPRegularLanguage]") {

	GIVEN("A number of simple matching expressions consist of a small number of matching operators") {
		using namespace SimpleMatcher;

		WHEN("Some simple inputs string are provided") {
			constexpr static string_view EmptyString = "";

			THEN("Inputs are matched accordingly returning the correct matching length") {
				CHECK(MatchAny::match("!") == 1u);
				CHECK(MatchLiteralA::match(StringA) == StringA.length());
				CHECK(MatchLiteralB::match(StringB) == StringB.length());

				CHECK(MatchAorB::match(StringA) == StringA.length());
				CHECK(MatchAorB::match(StringB) == StringB.length());
				CHECK(MatchUpper::match("C") == 1u);
				CHECK(MatchSymbol::match("+") == 1u);

				CHECK(MatchMaybeA::match(StringA) == StringA.length());
				CHECK(MatchMaybeA::match(StringB) == 0u);//0 match length is not the same as no match
				CHECK(MatchAtLeast2A::match("aaaa") == 4u);
				CHECK(Match5A::match("aaaaa") == 5u);

				CHECK(MatchAorBAlt::match(StringA) == StringA.length());
				CHECK(MatchAorBAlt::match(StringB) == StringB.length());

				CHECK(MatchABAny::match("ab*") == 3u);

				/* ---------------------- empty input --------------------- */
				CHECK_FALSE(MatchAny::match(EmptyString));
				CHECK_FALSE(MatchLiteralA::match(EmptyString));
				CHECK_FALSE(MatchAorB::match(EmptyString));
				CHECK(*MatchMaybeA::match(EmptyString) == 0u);
				CHECK_FALSE(MatchAorBAlt::match(EmptyString));
				CHECK_FALSE(MatchABAny::match(EmptyString));
			}

		}

	}

	GIVEN("Some production-quality matching expressions built from regular language operators") {
		
		WHEN("Some input string sequences are supplied to the finite state machine") {

			THEN("Inputs are matched by the expression, returning correct matching length") {
				//run every matcher example
				{
					using namespace IPv6Matcher;
					RUN_MATCHER_TEST;
				}
				{
					using namespace VariableDefinitionMatcher;
					RUN_MATCHER_TEST;
				}
			}

		}

	}

}