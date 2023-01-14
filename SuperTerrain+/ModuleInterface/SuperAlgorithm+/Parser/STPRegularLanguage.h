#pragma once
#ifndef _STP_REGULAR_LANGUAGE_H_
#define _STP_REGULAR_LANGUAGE_H_

#include <string_view>
#include <limits>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPRegularLanguage is a template library for compile-time finite state automata generation using regular language.
	 * This namespace defines a collection of regular language operator.
	*/
	namespace STPRegularLanguage {

		//Internal implementation for the regular language parser, you shouldn't need to call any function from this namespace.
		namespace STPDetail {
			
			/**
			 * @brief Match the sequence given an expression.
			 * @tparam Expr The expression.
			 * @param sequence The input string to be matched.
			 * After matching the input will be trimmed to remove matched substring.
			 * @param total_length An cumulative counter to add the current matching length to.
			 * @return True if there is a valid match for the given expression, false otherwise.
			*/
			template<class Expr>
			bool matchExpression(std::string_view&, size_t&) noexcept;

		}

		/**
		 * @brief Special operators used by the list matching operator.
		*/
		namespace STPListElement {

			/**
			 * @brief Add a single character to the matching list.
			 * @param C The character to be added.
			*/
			template<char C>
			struct Atomic { };

			/**
			 * @brief Add a range of character to the matching list.
			 * The range is based on ASCII, and close-end for both sides.
			 * @param First The first character in the range.
			 * @param Last The last character in the range.
			*/
			template<char First, char Last>
			struct Range {
			
				static_assert(First < Last, "The starting character should be strictly less than the ending character. "
					"If they are the same, consider using atomic list element.");

			};

			/**
			 * @brief Match character(s) not presenting in the list.
			 * @tparam L... The list elements should be excluded.
			*/
			template<class... L>
			struct Except {
			
				static_assert(sizeof...(L) > 0u, "The number of character class for an except list operator must be positive.");
			
			};

		}

		//A special matching length indicating no match.
		constexpr static size_t NoMatch = std::numeric_limits<size_t>::max();
		//Special value for matching unlimited number of maximum number of repetition.
		constexpr static size_t Unlimited = NoMatch;

		//This is the entry function for each operator.
		//Given a sequence, return the length of matching starting.
		//Return a special value *NoMatch* if there is no match.
#define DECLARE_REGLANG_MATCHER static size_t match(const std::string_view&) noexcept

		/**
		 * @brief Match any one character.
		*/
		struct Any {
		public:

			DECLARE_REGLANG_MATCHER;

		};

		/**
		 * @brief Match a group of string literal.
		 * @param L The compile-time string literal.
		*/
		template<const std::string_view& L>
		struct Literal {
		public:

			//The length of string literal.
			constexpr static size_t LiteralLength = L.length();

			static_assert(Literal::LiteralLength > 0u, "The matching literal should not be empty");

			DECLARE_REGLANG_MATCHER;

		};

		/**
		 * @brief Match a sequence of input character if any of the character is specified in the list.
		 * @tparam ...LE A collection of list element.
		 * @see STPListElement
		*/
		template<class... LE>
		struct List {
		private:

			static_assert(sizeof...(LE) > 0u, "The number of character class in a list operator must be positive.");

			/**
			 * @brief Match one list element.
			 * @tparam E The type of list element.
			*/
			template<class E>
			struct ElementSpecification;

			//Given a character input, search if the list element contains this character.
#define DECLARE_LIST_ELEMENT_CONTAINS static bool contains(char) noexcept

			//specialisation for each list element type
			template<char C>
			struct ElementSpecification<STPListElement::Atomic<C>> {
			public:

				DECLARE_LIST_ELEMENT_CONTAINS;
			
			};

			template<char First, char Last>
			struct ElementSpecification<STPListElement::Range<First, Last>> {
			public:

				DECLARE_LIST_ELEMENT_CONTAINS;

			};

			template<class... L>
			struct ElementSpecification<STPListElement::Except<L...>> {
			public:

				DECLARE_LIST_ELEMENT_CONTAINS;

			};

#undef DECLARE_LIST_ELEMENT_CONTAINS

		public:

			DECLARE_REGLANG_MATCHER;
		
		};

		/**
		 * @brief Match an expression for a repeated range of numbers of time.
		 * A match is considered if the number of repetition falls in the bound.
		 * The bound of repetition is close-end for both sides.
		 * @tparam Expr The expression to be repeated.
		 * @param Min The minimum number of encountering.
		 * @param Max The maximum number of encountering
		*/
		template<class Expr, size_t Min, size_t Max = Min>
		struct Repeat {
		public:

			static_assert(Min <= Max, "The minimum number of repetition should be no greater than the maximum");

			DECLARE_REGLANG_MATCHER;

		};

		/**
		 * @brief Match a sequence of expressions until the first successful matching is found.
		 * Match fails if none of the expression has a successful matching.
		 * This operator will create a NFA which needs to perform backtracking if a unsuccessful matching is encountered, which can be expensive.
		 * @tparam Expr... A sequence of expressions to be used for matching.
		*/
		template<class... Expr>
		struct Alternative {
		public:

			static_assert(sizeof...(Expr) > 0u, "The number of expression in an alternative operator must be positive.");

			DECLARE_REGLANG_MATCHER;
		
		};

		/**
		 * @brief Match a sequence of expression and concatenate the resulting matching length.
		 * A match is considered if and only if all expressions return a match.
		 * @tparam Expr... A sequence of expressions to be used for matching.
		*/
		template<class... Expr>
		struct Sequence {
		public:

			static_assert(sizeof...(Expr) > 0u, "The number of expression in a sequence operator must be positive.");

			DECLARE_REGLANG_MATCHER;
		
		};

#undef DECLARE_REGLANG_MATCHER
	}

}
#include "STPRegularLanguage.inl"
#endif//_STP_REGULAR_LANGUAGE_H_