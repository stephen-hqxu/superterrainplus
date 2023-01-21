//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

//SuperAlgorithm+/Parser
#include <SuperAlgorithm+/Parser/STPBasicStringAdaptor.h>
#include <SuperAlgorithm+/Parser/STPLexer.h>

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

//String
#include <string>
#include <string_view>

#include <array>
#include <limits>

using std::numeric_limits;
using std::string;
using std::string_view;
using std::array;

using namespace SuperTerrainPlus::STPAlgorithm;
namespace STPParserError = SuperTerrainPlus::STPException::STPParserError;

TEMPLATE_TEST_CASE("STPBasicStringAdaptor can convert between string and desired types",
	"[AlgorithmHost][STPBasicStringAdaptor]", string, string_view) {
	using namespace Catch::Matchers;
	typedef STPBasicStringAdaptor<TestType> CurrentSA;

	GIVEN("Some string representations of values that are not valid to be converted") {
		constexpr static char HelloWorld[] = "HelloWorld", OverflowChar[] = "257";

		WHEN("Creating string adaptors from these value") {
			const CurrentSA String1(HelloWorld), String2(OverflowChar);

			THEN("Then can be used as normal string") {
				CHECK((String1.to<string_view>() == string_view(HelloWorld)));
				CHECK((String1.to<string>() == string(HelloWorld)));

				CHECK((*String1 == TestType(HelloWorld)));
				CHECK((*String2 == TestType(OverflowChar)));

				AND_THEN("They cannot be converted to desired types") {
					CHECK_THROWS_WITH(String1.to<unsigned int>(), ContainsSubstring(HelloWorld));
					CHECK_THROWS_WITH(String2.to<unsigned char>(), ContainsSubstring(OverflowChar));
				}

			}

		}

	}

	GIVEN("Some string representations of the values that can be converted") {
		constexpr static char UShortMax[] = "65535u", DoublePi[] = "3.1415", BoolFalse[] = "false";

		WHEN("The string adaptors are constructed with these values") {
			const CurrentSA String1(UShortMax), String2(DoublePi), String3(BoolFalse);

			THEN("The corresponded values can be converted from the adaptors") {
				CHECK(String1.to<unsigned short>() == numeric_limits<unsigned short>::max());
				CHECK_THAT(String2.to<double>(), Catch::Matchers::WithinAbs(3.1415, 1e-5));
				CHECK_FALSE(String3.to<bool>());
			}

			AND_WHEN("The string adaptors are copied or moved") {
				const CurrentSA StringCpy(String1), StringMov(std::move(String3));

				THEN("They functions exactly the same as the original copy") {
					CHECK((*StringCpy == *String1));
					CHECK_FALSE(StringMov.to<bool>());
				}

			}

		}

	}

}

namespace RL = STPRegularLanguage;
namespace CC = RL::STPCharacterClass;

using EOS = STPLexical::EndOfSequence;

namespace {
	using RL::STPQuantifier::StrictMany;

	constexpr string_view Terminator = ";", Space = " ", Newline = "\n",
		Right = ">", Left = "<", SymbA = "a", SymbB = "b";

	using OnlyA = StrictMany<RL::Literal<SymbA>>;
	using OnlyB = StrictMany<RL::Literal<SymbB>>;
	using AorB = StrictMany<CC::Class<CC::Atomic<'a'>, CC::Atomic<'b'>>>;

	//test different actions
	STP_LEXER_CREATE_TOKEN_EXPRESSION(WithinData, 7777u, Collect, CC::Class<CC::Except<CC::Atomic<';'>>>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DataLineEnd, 8888u, Consume, RL::Literal<Terminator>);

	STP_LEXER_CREATE_TOKEN_EXPRESSION(SkipSpace, 16666u, Discard, RL::Literal<Space>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(SimpleSymbol, 17777u, Consume, AorB);

	STP_LEXER_CREATE_LEXICAL_STATE(StateData, 66u, WithinData, DataLineEnd);
	STP_LEXER_CREATE_LEXICAL_STATE(StateSymbol, 77u, SkipSpace, SimpleSymbol);

	//test lexical state switch
	STP_LEXER_CREATE_TOKEN_EXPRESSION(CharA, 998u, Consume, OnlyA);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(CharB, 999u, Consume, OnlyB);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(ToStateB, 13u, Discard, RL::Literal<Right>, 13u);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(ToStateA, 14u, Discard, RL::Literal<Left>, 0u);

	STP_LEXER_CREATE_LEXICAL_STATE(StateA, 0u, CharA, ToStateB);
	STP_LEXER_CREATE_LEXICAL_STATE(StateB, 13u, CharB, ToStateA);
}

SCENARIO("STPLexer can perform basic lexical operations", "[AlgorithmHost][STPLexer]") {

	GIVEN("A lexer defined with a state with some token expressions") {
		constexpr static string_view LexerName = "Test Lexical Action";
		typedef STPLexer<StateData> DataLexer;
		typedef STPLexer<StateSymbol> SymbolLexer;

		WHEN("A string source with invalid token by definition of the lexer") {
			//there is no ending symbol
			constexpr static char InvalidTokenSource[] = "chicken", BrokenSourceName[] = "TheBrokenCode.bad";
			DataLexer Lexer(InvalidTokenSource, LexerName, BrokenSourceName);

			THEN("The lexer fails due to appearance of a undefined token") {
				REQUIRE_THROWS_AS(Lexer.expect<DataLineEnd>(), STPParserError::STPInvalidSyntax);
			}

		}

		WHEN("Simple valid inputs are supplied") {
			constexpr static string_view Data = "hello, world!;", Symbol = "aa ab ba bb",
				DataName = "Data.txt", SymbolName = "SymbolList.txt";
			DataLexer LexerA(Data, LexerName, DataName);
			SymbolLexer LexerB(Symbol, LexerName, SymbolName);

			THEN("The lexer can analyse the input and provide the token stream with the `expect` expression") {
				//test each lexer
				CHECK((**LexerA.expect<DataLineEnd>() == Data));
				
				//---------------------------------
				constexpr static array<string_view, 4u> SymbolToken = { "aa", "ab", "ba", "bb" };
				for (const auto& token : SymbolToken) {
					CHECK((**LexerB.expect<SimpleSymbol>() == token));
				}

				AND_THEN("The lexer only returns null token when the source reaches the end") {
					//loop it for some iterations for robustness
					for (int i = 0; i < 4; i++) {
						CHECK_NOTHROW(LexerA.expect<EOS>());
						CHECK_NOTHROW(LexerB.expect<EOS>());
					}
				}

			}

			AND_WHEN("The lexer has been provided with empty `expect` expression") {

				THEN("The lexer always returns the valid token") {
					CHECK(LexerA.expect() == DataLineEnd {});
				}

			}

		}

	}

	GIVEN("A lexer defined multiple states with tokens that switch in-between these states") {
		constexpr static string_view LexerName = "Test Lexical State";
		typedef STPLexer<StateA, StateB> SymbolSwitchLexer;

		WHEN("Slightly more complex inputs are supplied") {
			constexpr static string_view Input = "aaaaaa>bbb<a", InputName = "SymbolWithStateSwitch.txt";
			SymbolSwitchLexer Lexer(Input, LexerName, InputName);

			THEN("The lexer can still analyse with the correct state switching") {
				//start from the first state
				CHECK(Lexer.CurrentState == StateA::LexicalStateID);
				CHECK((**Lexer.expect<CharA>() == "aaaaaa"));

				//state switch symbols should be ignored
				CHECK_NOTHROW((**Lexer.expect<CharB>() == "bbb"));
				//state should have been switched
				CHECK(Lexer.CurrentState == StateB::LexicalStateID);

				//now switch back
				CHECK((**Lexer.expect<CharA>() == "a"));
				CHECK(Lexer.CurrentState == StateA::LexicalStateID);

				CHECK_NOTHROW(Lexer.expect<EOS>());
			}

			AND_THEN("The lexer cannot match tokens that are not in the same state") {
				constexpr static string_view BadInput = "bb";
				SymbolSwitchLexer LexerBad(BadInput, LexerName, InputName);

				CHECK_THROWS_AS(LexerBad.expect(), STPParserError::STPInvalidSyntax);
			}

		}

	}

}