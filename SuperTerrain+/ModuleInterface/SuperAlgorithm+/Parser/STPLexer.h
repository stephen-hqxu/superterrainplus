#pragma once
#ifndef _STP_LEXER_H_
#define _STP_LEXER_H_

#include "STPBasicStringAdaptor.h"
#include "STPRegularLanguage.h"

//Container
#include <array>
#include <tuple>
//String
#include <string>
#include <string_view>
//Utility
#include <limits>
#include <utility>
#include <type_traits>

/*
Create a new token expression for token matching within a state.
Once a token is matched, the state will be switched to a new state as indicated.
---
NAME: The name for this token expression.
TEID: Assign a unique token expression ID.
ACTION: Specifies an action to perform if this expression matches a token.
EXPR: Specifies a token matching expression defined in regular language.
NEXT_SID: Specifies the state ID to be switched to once a matching is found for this expression.
*/
#define STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(NAME, TEID, ACTION, EXPR, NEXT_SID) struct NAME{ \
	constexpr static std::string_view Representation = #NAME; \
	constexpr static SuperTerrainPlus::STPAlgorithm::STPLexical::STPTokenID LexicalTokenExpressionID = TEID; \
	constexpr static SuperTerrainPlus::STPAlgorithm::STPLexical::STPAction LexicalAction = \
		SuperTerrainPlus::STPAlgorithm::STPLexical::STPAction::ACTION; \
	using TokenExpression = EXPR; \
	constexpr static SuperTerrainPlus::STPAlgorithm::STPLexical::STPStateID NextLexicalState = NEXT_SID; \
}
/*
Similarly; but the state remains the current state once a matching is found.
*/
#define STP_LEXER_CREATE_TOKEN_EXPRESSION(NAME, TEID, ACTION, EXPR) \
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(NAME, TEID, ACTION, EXPR, \
		SuperTerrainPlus::STPAlgorithm::STPLexical::NullStateID)
/* ---------------------------------------------------------------------------- */

/*
Create a new lexical state.
---
NAME: The name for this lexical state.
SID: Assign a unique state ID.
...: Specifies a collection of lexical token matching expressions in this lexical state.
*/
#define STP_LEXER_CREATE_LEXICAL_STATE(NAME, SID, ...) struct NAME { \
	constexpr static std::string_view Representation = #NAME; \
	constexpr static SuperTerrainPlus::STPAlgorithm::STPLexical::STPStateID LexicalStateID = SID; \
	using TokenExpressionCollection = std::tuple<__VA_ARGS__>; \
}

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPLexical provides some definitions and tools for building and manipulating on STPLexer.
	 * @see STPLexer
	*/
	namespace STPLexical {

		//An ID to uniquely identify a lexical state.
		typedef unsigned char STPStateID;
		//An ID to uniquely identify a token generated by a matched token expression.
		typedef unsigned short STPTokenID;

		//A special value indicating a lexical state that is not valid.
		inline constexpr STPStateID NullStateID = std::numeric_limits<STPStateID>::max();
		//A special value indicating a token that is not defined.
		inline constexpr STPTokenID NullTokenID = std::numeric_limits<STPTokenID>::max(),
			//A special token value for the end of input token.
			EndOfSequenceTokenID = NullTokenID - 1u;

		//A special token to indicate the input sequence has reached the end.
		enum struct EndOfSequence : unsigned char {};

		/**
		 * @brief STPAction specifies the type of action to perform when a match happens on a token.
		*/
		enum class STPAction : unsigned char {
			//Simply throw away matched string.
			Discard = 0xAAu,
			//Continue whatever the next state is, taking the matched string along.
			//This string will be a prefix of the new matched string.
			Collect = 0xBBu,
			//Create a token using the matched string and return the token.
			Consume = 0xCCu
		};

	}

	/**
	 * @brief STPLexer, whose specification is inspired by JavaCC,
	 * is a simple general-purpose lexer that generates an array of tokens for a given string input.
	 * A lexer is user-defined with a collection of lexical states, each of them contain a number of lexical token expressions.
	 * Each token expression provide some different types of lexical action to perform when a match appears.
	 * 
	 * All token expressions in a state are considered as potential candidates for matching a token from the input.
	 * The lexer consumes the maximum number of characters from the input possible that matches one of these expression.
	 * If there are multiple longest match of the same length, the lexer chooses the token that matches the one with the earliest order of
	 * occurrence in the collection of token expression in the current state.
	 * @tparam LexState... Specify an parameter of user-defined lexical states.
	 * The first state will be the default state for which the lexer is operating on upon initialisation.
	 * @see STPLexical
	*/
	template<class... LexState>
	class STPLexer {
	private:

		//all lexical states defined in this lexer
		using LexicalStateCollection = std::tuple<LexState...>;

		/* ----------------------- say hello to all type-level programming below ------------------------ */
		template<typename IDType, IDType I, IDType... Is>
		constexpr static bool isAllIDUnique() noexcept {
			//we need to ensure all tokens have unique ID
			if constexpr (sizeof...(Is) == 0u) {
				//base case
				return true;
			} else {
				//recursive case
				return ((I != Is) && ...) && STPLexer::isAllIDUnique<IDType, Is...>();
			}
		}
		//call the comparison function using the sequence of ID
		//also make sure the id does not equal to the special id
		template<typename IDType, IDType... I>
		constexpr static bool isIDValid(std::integer_sequence<IDType, I...>, const IDType reserve_start) noexcept {
			return ((I < reserve_start) && ...) && STPLexer::isAllIDUnique<IDType, I...>();
		}

		//utility to unpack token expression tuple
		template<class>
		struct STPTokenExpressionUtility;

		template<class... TExpr>
		struct STPTokenExpressionUtility<std::tuple<TExpr...>> {
			//convert tuple of token expression classes to sequence of their IDs
			using ExtractIDSequence = std::integer_sequence<STPLexical::STPTokenID, TExpr::LexicalTokenExpressionID...>;

			/**
			 * @brief Match an input character sequence with all token matching expressions.
			 * @param sequence An input sequence to be matched.
			 * @return An array of matching length in order of declaration of token expression.
			*/
			static auto match(const std::string_view& sequence) noexcept {
				return std::array { TExpr::TokenExpression::match(sequence)... };
			}
		};

		/* ---------------------------------------------------------------------------------- */
		//check if state IDs are unique
		using LexicalStateCollectionID = std::integer_sequence<STPLexical::STPStateID, LexState::LexicalStateID...>;
		static_assert(STPLexer::isIDValid(LexicalStateCollectionID {}, STPLexical::NullStateID),
			"There are lexical states defined in this lexer whose IDs are not unique and/or equal to any reserved value");
		
		//check if token expression IDs are unique
		//this is a bit more tricker, we need join all inner tuple of token expression classes together
		using LexicalTokenExpressionCollectionID = typename STPTokenExpressionUtility<decltype(std::tuple_cat(
			std::declval<typename LexState::TokenExpressionCollection>()...))>::ExtractIDSequence;
		static_assert(STPLexer::isIDValid(LexicalTokenExpressionCollectionID {}, std::min(STPLexical::NullTokenID, STPLexical::EndOfSequenceTokenID)),
			"There are lexical token expressions defined in this lexer whose IDs are not unique and/or equal to any reserved value");

		/* -------------------------------------------------------------------------------------------------- */
		//store a map entry
		template<STPLexical::STPStateID V, size_t I>
		struct STPStateIDMapEntry {
			//state ID
			constexpr static auto ID = V;
			//index in the map for this state ID
			constexpr static auto Index = I;
		};

		//build a compile-time reverse lookup table for state ID and index into the state collection
		template<class, class>
		struct buildStateIDMap;

		template<typename IDType, IDType... V, size_t... I>
		struct buildStateIDMap<std::integer_sequence<IDType, V...>, std::index_sequence<I...>> {
			using Map = std::tuple<STPStateIDMapEntry<V, I>...>;
		};
		//convert from lexical state ID to index in the parameter pack
		using LexicalStateCollectionIDMap =
			typename buildStateIDMap<LexicalStateCollectionID, std::make_index_sequence<sizeof...(LexState)>>::Map;

		/**
		 * @brief Convert from lexical state ID to the index in the collection.
		 * @param E All state entries in the map.
		 * @param state The state ID to be converted. The behaviour is undefined if the state ID is not an ID belongs
		 * to a state defined in this lexer.
		 * @return The lexical state index.
		*/
		template<class... E>
		constexpr static size_t toStateIndex(const STPLexical::STPStateID state, std::tuple<E...>) noexcept {
			//compile-time linear search for the index
			size_t index = std::numeric_limits<size_t>::max();
			//search using fold expression; make it volatile to prevent compiler from optimising it away
			//this expression should always be true
			[[maybe_unused]] const volatile bool isValidStateID = ((index = E::Index, state == E::ID) || ...);
			return index;
		}

		/* --------------------------------------------------------------------------------------------- */
		//extract token expressions from one state into a more accessible data structure
		template<class... T>
		constexpr static auto getTokenExpressionData(std::tuple<T...>) noexcept {
			return std::array { std::make_tuple(
				STPLexer::getTokenName<T>(),
				T::LexicalTokenExpressionID,
				T::LexicalAction,
				T::NextLexicalState
			)... };
		}

		//instantiate state matching function for all states
		template<typename IDType, IDType... S>
		constexpr static auto instantiateStateMatcher(std::integer_sequence<IDType, S...>) noexcept {
			return std::array { &STPLexer::nextAtState<S>... };
		}

		/* ---------------------------------------------------------------------------------------------- */

	public:

		/**
		 * @brief STPToken specifies a lexer token matched from the input string.
		*/
		struct STPToken {
		private:

			friend class STPLexer;

			//the name for some special tokens
			constexpr static std::string_view Nameless = "InvalidToken", EOSName = "EOS";

			//check if the token is a EOS token
			template<class T>
			constexpr static bool isEOS = std::is_same_v<T, STPLexical::EndOfSequence>;

		public:

			//the name given by user from matched token expression
			const std::string_view& Name;
			//the ID of the token recognised.
			const STPLexical::STPTokenID TokenID;
			//the string representation of this token.
			const STPStringViewAdaptor Lexeme;

			/**
			 * @brief Create a default token, which is invalid.
			*/
			constexpr STPToken() noexcept;

			/**
			 * @brief Create a new token.
			 * @param name The name of the token.
			 * This is only for debugging purposes.
			 * @param id The ID of this token.
			 * @param lexeme The string view of the lexeme captured by this token.
			*/
			constexpr STPToken(const std::string_view&, STPLexical::STPTokenID, std::string_view) noexcept;

			~STPToken() = default;

			/**
			 * @brief Test if the token captured by this instance is valid.
			 * @return True if the token is valid.
			*/
			explicit operator bool() const noexcept;

			//Get the underlying lexeme object reference.
			const STPStringViewAdaptor& operator*() const noexcept;
			//Get the underlying lexeme object pointer.
			const STPStringViewAdaptor* operator->() const noexcept;

			/**
			 * @brief Compare the equality of this token and the user-defined token expression, based on the token ID.
			 * @tparam TokExpr The lexer token expression to be compared to.
			 * @param right The lexer token expression instance to be compared.
			 * @return True if this token has the same token ID as the lexer token expression.
			*/
			template<class TokExpr>
			bool operator==(TokExpr) const noexcept;
			//@see operator==()
			template<class TokExpr>
			bool operator!=(TokExpr) const noexcept;

		};

	private:

		//A string input containing a sequence of null-terminated characters to be parsed.
		std::string_view Sequence;
		//The currently processed line and number of character in a row.
		size_t Line, Character;

		/**
		 * @brief Obtain the name of a token.
		 * @tparam TokExpr The token to get the name from.
		 * @return The name of the token.
		 * The lifetime of the variable is static.
		*/
		template<class TokExpr>
		constexpr static const std::string_view& getTokenName() noexcept;

		/**
		 * @brief Throw a syntax error due to unexpected token.
		 * @tparam ExpTokExpr... Given a list of expected tokens
		 * @param got_token The token actually got from the source.
		*/
		template<class... ExpTokExpr>
		[[noreturn]] void throwUnexpectedTokenError(const STPToken&) const;

		/**
		 * @brief Remove the a number character from the string sequence.
		 * This function automatically corrects the line and character number if the popped sequence contains line feed.
		 * @param count The number of character to be popped.
		 * @return The lexeme of the popped sequence.
		*/
		std::string_view pop(size_t);

		/**
		 * @brief Match the next token in a given state.
		 * @param S The state ID to be operating within.
		 * @param sequence The sequence of characters to be matched.
		 * This is usually a subsequence of the lexer sequence.
		 * @return The matching length and the pointer to data of matched token expression.
		 * The pointer has static storage duration.
		 * If there is no match, return NoMatch and null.
		*/
		template<STPLexical::STPStateID S>
		static auto nextAtState(std::string_view) noexcept;

		/**
		 * @brief Find the next token.
		 * If the end of string is reached, this function always returns null token regardless of the number of time it is called.
		 * @return The next matched token.
		 * Returns empty token if no match is found.
		*/
		STPToken next();

	public:

		//The name of the lexer and parser that uses this lexer framework, for debugging purposes.
		const std::string_view LexerName;
		//A source name for debugging purposes.
		const std::string_view SourceName;

		//The current lexical state the lexer is operating at.
		//This state will be initialised as the state appears as the first lexical state in the state collection.
		//State can be modified manually, or by matched token.
		STPLexical::STPStateID CurrentState;

		/**
		 * @brief Initialise a lexer instance.
		 * @param input The pointer to the source code to be parsed. The source string should be null-terminated.
		 * The lexer does not own the string, the memory of the original string should be managed by the user, 
		 * until the current instance and all returned view of string from the current instance to the user are destroyed.
		 * @param lexer_name The name of the lexer or parser using this general-purpose lexer.
		 * @param source_name The name of the source file.
		 * This is optional, just to make debugging easier.
		*/
		STPLexer(std::string_view, std::string_view, std::string_view = "<unknown source name>") noexcept;

		~STPLexer() = default;

		//TODO: use source location in C++20, this function throws exception with source information within this function.
		//we want to capture the source of the lexer who calls this function.
		/**
		 * @brief Throws a syntax error.
		 * @param message The description related to the error.
		 * @param error_title A summary of the error.
		*/
		[[noreturn]] void throwSyntaxError(const std::string&, const char*) const;

		/**
		 * @brief Expect the next token to be a given list of possible tokens.
		 * @tparam ExpTokExpr... The collection of lexical token expression expected, for which it will be used to compare against any matched token.
		 * If no expected token is given, the matching always passes and returns the next valid token found.
		 * @return The first matched token.
		 * If the next token does not match any of the expected, or an invalid token is encountered, an exception will be generated.
		*/
		template<class... ExpTokExpr>
		STPToken expect();
	
	};

}
#include "STPLexer.inl"
#endif//_STP_LEXER_H_