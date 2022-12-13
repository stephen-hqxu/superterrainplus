//Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>

//SuperAlgorithm+/Parser
#include <SuperAlgorithm+/Parser/STPBasicStringAdaptor.h>
#include <SuperAlgorithm+/Parser/STPLexer.h>

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

//String
#include <string>
#include <string_view>
#include <cctype>

#include <limits>

using std::string;
using std::string_view;
using std::tuple;
using std::numeric_limits;

using namespace SuperTerrainPlus::STPAlgorithm;
namespace STPParserError = SuperTerrainPlus::STPException::STPParserError;

TEMPLATE_TEST_CASE("STPBasicStringAdaptor can convert between string and desired types",
	"[AlgorithmHost][STPBasicStringAdaptor]", string, string_view) {
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
					CHECK_THROWS_AS(String1.to<unsigned int>(), STPParserError::STPSemanticError);
					CHECK_THROWS_AS(String2.to<unsigned char>(), STPParserError::STPSemanticError);
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
				CHECK(String3.to<bool>() == false);
			}

		}

	}

}

//define function token
namespace {
	template<class Func>
	inline size_t simpleMatch(Func&& f, const char* const sequence) {
		size_t pos = 0u;
		while (std::forward<Func>(f)(sequence[pos])) {
			pos++;
		}
		return pos;
	}

	STP_LEXER_DECLARE_FUNCTION_TOKEN(PureStringToken, 6666u, "Pure String");
	STP_LEXER_DECLARE_FUNCTION_TOKEN(PureNumberToken, 8888u, "Pure Number");
}

STP_LEXER_DEFINE_FUNCTION_TOKEN(PureStringToken) {
	return simpleMatch(isalpha, sequence);
}

STP_LEXER_DEFINE_FUNCTION_TOKEN(PureNumberToken) {
	return simpleMatch(isdigit, sequence);
}

SCENARIO("STPLexer can tokenise a string based on application-defined tokens", "[AlgorithmHost][STPLexer]") {
	namespace LT = STPLexerToken;
	typedef tuple<LT::Equal, LT::Comma, LT::Semicolon, LT::Null> TestAtomToken;
	typedef tuple<PureStringToken, PureNumberToken> TestFuncToken;
	//define our lexer
	typedef STPLexer<TestAtomToken, TestFuncToken> SimpleVariableLexer;
	constexpr static string_view LexerName = "Test Simple Variable Lexer";

	GIVEN("A string source with invalid token by definition of the lexer") {
		//we don't have a colon token
		constexpr static string_view InvalidTokenSource = "0:chicken", BrokenSourceName = "TheBrokenCode.bad";
		SimpleVariableLexer Lexer(InvalidTokenSource, LexerName, BrokenSourceName);

		WHEN("The lexer is attempting to analyse it") {

			THEN("The lexer fails due to appearance of a undefined token") {
				REQUIRE(Lexer.expect<PureNumberToken>());
				REQUIRE_THROWS_AS(Lexer.expect<PureStringToken>(), STPParserError::STPInvalidSyntax);
			}

		}

	}

	GIVEN("A string source to be parsed and a defined lexer") {
		constexpr static string_view Source = "int myNumber = 3;\nshort numA=1,numB =4, numC= 1 ;",
			SourceName = "TheSourceOfPi.magic";
		SimpleVariableLexer Lexer(Source, LexerName, SourceName);

		THEN("The information of the lexer and source can be retrieved") {
			CHECK((Lexer.LexerName == LexerName));
			CHECK((Lexer.SourceName == SourceName));
		}

		WHEN("The application is going to conduct lexicological analysis") {
			constexpr static unsigned int ExpectedOutput = 3141u;

			THEN("The lexer can outputs the correct stream of tokens") {
				STPStringAdaptor ParsedOutput;
				ParsedOutput->reserve(4u);
				//parse variable
				const auto parseKeyValPair = [&output = ParsedOutput, &lexer = Lexer]() -> void {
					lexer.expect<PureStringToken>();
					lexer.expect<LT::Equal>();
					output->append(lexer.expect<PureNumberToken>()->to<string>());
				};

				while (true) {
					if (Lexer.expect<PureStringToken, LT::Null>() == LT::Null {}) {
						//end of the string source
						break;
					}
					parseKeyValPair();

					while (true) {
						if (Lexer.expect<LT::Semicolon, LT::Comma>() == LT::Semicolon {}) {
							//next variable declaration start
							break;
						}
						parseKeyValPair();
					}
				}

				REQUIRE(ParsedOutput.to<unsigned int>() == ExpectedOutput);
			}

		}

	}

}