#pragma once
#ifndef _STP_LEXER_H_
#define _STP_LEXER_H_

#include "STPBasicStringAdaptor.h"

//Container
#include <tuple>
//String
#include <string>
#include <string_view>
//Utility
#include <limits>
#include <utility>

//Declare a new function token that matches using a function, given the function name and the token ID.
//The symbol value is the name for this token and is purely for debugging purposes as a function token, it should be a compile-time string literal,
//and it can be printed to the console when any lexer error is encountered.
#define STP_LEXER_DECLARE_FUNCTION_TOKEN(NAME, TOK_ID, SYMB) struct NAME { \
public: \
	constexpr static SuperTerrainPlus::STPAlgorithm::STPLexerToken::STPTokenID ID = TOK_ID; \
	constexpr static char Representation[] = "<" SYMB ">"; \
	size_t operator()(const char*) const; \
};
//Define the function token matching function, the name should also include any namespace, if defined within any.
//The function should return the length of string into the sequence that matches.
//The length of the substring should not go past the null terminator of the source string, otherwise UB.
#define STP_LEXER_DEFINE_FUNCTION_TOKEN(NAME) size_t NAME::operator()(const char* const sequence) const

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPLexerToken defines a list of possible tokens to be used by the lexer.
	*/
	namespace STPLexerToken {

		//an ID to uniquely identify a token
		typedef unsigned short STPTokenID;
		//indicate a token that is not defined
		constexpr static STPTokenID NullTokenID = std::numeric_limits<STPTokenID>::max();

//define a new atomic token that only contains a single character
//the symbol will be used as a equality comparator to match the character
#define DEF_ATOM_TOKEN(NAME, SYMB) struct NAME { \
public: \
	constexpr static STPTokenID ID = static_cast<STPTokenID>(SYMB); \
	constexpr static char Character = SYMB; \
	constexpr static char Representation[] = #SYMB; \
	}
		//control
		DEF_ATOM_TOKEN(Null, '\0');

		//symbols
		DEF_ATOM_TOKEN(Hash, '#');
		DEF_ATOM_TOKEN(Comma, ',');
		DEF_ATOM_TOKEN(Colon, ':');
		DEF_ATOM_TOKEN(Semicolon, ';');
		DEF_ATOM_TOKEN(QuestionMark, '?');

		//arithmetic
		DEF_ATOM_TOKEN(Minus, '-');
		DEF_ATOM_TOKEN(Equal, '=');
		DEF_ATOM_TOKEN(GreaterThan, '>');

		//brackets
		DEF_ATOM_TOKEN(LeftRound, '(');
		DEF_ATOM_TOKEN(RightRound, ')');
		DEF_ATOM_TOKEN(LeftSquare, '[');
		DEF_ATOM_TOKEN(RightSquare, ']');
		DEF_ATOM_TOKEN(LeftCurly, '{');
		DEF_ATOM_TOKEN(RightCurly, '}');

#undef DEF_ATOM_TOKEN
	}

	/**
	 * @brief STPLexer is a simple general-purpose lexer that generates an array of tokens for a given string input.
	 * It can be used to create custom parsers.
	 * @tparam TokA The pack of tokens for the first type.
	 * @tparam TokB The pack of tokens for the second type.
	*/
	template<class TokA, class TokB>
	class STPLexer;

#define LEXER_TEMPLATE template<class... AtomToken, class... FuncToken>
#define LEXER_NAME STPLexer<std::tuple<AtomToken...>, std::tuple<FuncToken...>>

	/**
	 * @brief A specialisation of the general-purpose lexer that takes two different types of tokens.
	 * This lexer matches atomic tokens first before moving on to function token matching.
	 * @see STPLexer
	 * @tparam AtomToken... A list of token that only contains a single atomic character.
	 * @tparam FuncToken... A list of token that matches using an application-defined function.
	 * Function tokens are more expensive, so they are called only when none of the atomic tokens are matched.
	*/
	LEXER_TEMPLATE
	class LEXER_NAME {
	private:

		//template argument check in compile-time
		template<STPLexerToken::STPTokenID I, STPLexerToken::STPTokenID... Is>
		constexpr static bool isAllIDUnique() {
			//we need to ensure all tokens have unique ID
			if constexpr (sizeof...(Is) == 0u) {
				//base case
				return true;
			} else {
				//recursive case
				return ((I != Is) && ...) && STPLexer::isAllIDUnique<Is...>();
			}
		}
		//put all token IDs to an integer sequence
		template<STPLexerToken::STPTokenID... I>
		constexpr static bool extractAllID(std::integer_sequence<STPLexerToken::STPTokenID, I...>) {
			return STPLexer::isAllIDUnique<I...>();
		}
		//ensure uniqueness of all token IDs
		static_assert(extractAllID(std::integer_sequence<STPLexerToken::STPTokenID, AtomToken::ID..., FuncToken::ID...> {}),
			"The lexer is ill-formed since there are tokens defined in this lexer whose IDs are not unique");

	public:

		//the number of atomic token
		constexpr static size_t AtomTokenCount = sizeof...(AtomToken);
		//the number of function token
		constexpr static size_t FuncTokenCount = sizeof...(FuncToken);

		/**
		 * @brief STPToken specifies a lexer token parsed from the input string.
		*/
		struct STPToken {
		public:

			//the ID of the token recognised
			const STPLexerToken::STPTokenID TokenID;
			//the string representation of this token
			const STPStringViewAdaptor Lexeme;

			/**
			 * @brief Create a default token, which is invalid.
			*/
			STPToken() noexcept;

			/**
			 * @brief Create a new token.
			 * @param id The ID of this token.
			 * @param lexeme The string view of the lexeme captured by this token.
			*/
			STPToken(STPLexerToken::STPTokenID, std::string_view) noexcept;

			~STPToken() = default;

			/**
			 * @brief Test if the token captured by this instance is valid.
			 * @return True if the token is valid.
			*/
			explicit operator bool() const noexcept;

			//get the underlying lexeme object reference
			const STPStringViewAdaptor& operator*() const noexcept;
			//get the underlying lexeme object pointer
			const STPStringViewAdaptor* operator->() const noexcept;

			/**
			 * @brief Compare the equality of the token and the lexer token, based on the token ID.
			 * @tparam Tok The lexer token to be compared to.
			 * @param right The lexer token to be compared.
			 * @return True if this token has the same token ID as the lexer token.
			*/
			template<class Tok>
			bool operator==(Tok&&) const noexcept;
			//@see operator==()
			template<class Tok>
			bool operator!=(Tok&&) const noexcept;

		};

	private:

		//A string input containing a sequence of null-terminated characters to be parsed.
		const char* Sequence;
		//The currently processed line and number of character in a row.
		size_t Line, Character;

		/**
		 * @brief Throw a syntax error due to unexpected token.
		 * @tparam ExpTok... Given a list of expected tokens
		 * @param got_token The token actually got from the source.
		*/
		template<class... ExpTok>
		[[noreturn]] void throwUnexpectedTokenError(const STPToken&) const;

		/**
		 * @brief Peek at the first character in the string sequence.
		 * @return The first character in the string sequence.
		*/
		char peek() const noexcept;

		/**
		 * @brief Remove the a number character from the string sequence.
		 * It is a undefined behaviour to pop from an empty string.
		 * @param count The number of character to be popped.
		*/
		void pop(size_t = 1u) noexcept;

		/**
		 * @brief Create a token and pop the characters from the current sequence.
		 * @param id The ID of the token.
		 * @param length The length of the sequence from the current sequence.
		 * @return A token.
		*/
		STPToken createToken(STPLexerToken::STPTokenID, size_t = 1u) noexcept;

		/**
		 * @brief Correct the lexer stats, such as line and character number.
		 * Use this when the token lexeme contains a long string and may potentially contain newline character.
		 * @param token The input token based on which will be adjusted accordingly.
		*/
		void correctStats(const STPToken&) noexcept;

		/**
		 * @brief Attempt to find the next atomic token.
		 * @return The next atomic token.
		 * Also returns an empty token if no match is found.
		*/
		STPToken nextAtomicToken() noexcept;

		/**
		 * @brief Attempt to find the next functional token.
		 * @tparam First The current function token.
		 * @tparam ...Rest The rest of the functional tokens.
		 * @return The next functional token.
		 * Or an empty token if no match is found.
		*/
		template<class First, class... Rest>
		STPToken nextFunctionToken() noexcept;

		/**
		 * @brief Find the next token.
		 * @return The next matched token.
		 * Returns empty token if no match is found.
		*/
		STPToken next() noexcept;

	public:

		//The name of the lexer and parser that uses this lexer framework, for debugging purposes.
		const std::string_view LexerName;
		//A source name for debugging purposes.
		const std::string_view SourceName;

		/**
		 * @brief Initialise the double-token lexer instance.
		 * @param input The pointer to the source code to be parsed. The source string should be null-terminated.
		 * The lexer does not own the string, the memory of the original string should be managed by the user, 
		 * until the current instance and all returned view of string from the current instance to the user are destroyed.
		 * @param lexer_name The name of the lexer or parser using this general-purpose lexer.
		 * @param source_name The name of the source file.
		 * This is optional, just to make debugging easier.
		*/
		STPLexer(const char*, const std::string_view&, const std::string_view& = "<unknown source name>") noexcept;

		~STPLexer() = default;

		/**
		 * @brief Throws a syntax error.
		 * @param message The description related to the error.
		 * @param error_title A summary of the error.
		*/
		[[noreturn]] void throwSyntaxError(const std::string&, const char*) const;

		/**
		 * @brief Expect the next token to be a given list of possible tokens.
		 * @tparam ExpTok... The collection of token expected.
		 * If not expected token is given, the matching always passes and returns the next valid token found.
		 * @return The first matched token.
		 * If the next token does not match any of the expected, an exception will be generated.
		*/
		template<class... ExpTok>
		STPToken expect();

	};

}
#include "STPLexer.inl"

#undef LEXER_TEMPLATE
#undef LEXER_NAME
#endif//_STP_LEXER_H_