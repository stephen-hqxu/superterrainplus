#pragma once
#ifndef _STP_REGULAR_LANGUAGE_H_
#define _STP_REGULAR_LANGUAGE_H_

#include <string_view>
#include <limits>
#include <optional>

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

		//The length of matched substring, or null if fail to match any.
		typedef std::optional<size_t> STPMatchLength;

		//This is the entry function for each operator.
		//Given a sequence, return the length of matching starting.
		//Return a special value *NoMatch* if there is no match.
#define DECLARE_REGLANG_MATCHER static STPMatchLength match(const std::string_view&) noexcept

		/**
		 * @brief STPCharacterClass matches any character that appears in the class.
		*/
		namespace STPCharacterClass {

			/**
			 * @brief Add a single character to the character class.
			 * @param C The character to be added.
			*/
			template<char C>
			struct Atomic { };

			/**
			 * @brief Add a range of character to the character class.
			 * The range is based on ASCII, and close-end for both sides.
			 * @param First The first character in the range.
			 * @param Last The last character in the range.
			*/
			template<char First, char Last>
			struct Range {
			
				static_assert(First < Last, "The starting character should be strictly less than the ending character. "
					"If they are the same, consider using atomic class member.");

			};

			/**
			 * @brief Add character(s) to the character class, that are complement to all character class members provided.
			 * This essentially excludes all characters that appears.
			 * @tparam C... A number of character class members to be excluded.
			*/
			template<class... C>
			struct Except {
			
				static_assert(sizeof...(C) > 0u, "The number of characters to be excluded must be positive.");
			
			};

			/**
			 * @brief Match a character if any of the character is specified in the character class.
			 * @tparam CM... A collection of character class member.
			*/
			template<class... CM>
			struct Class {
			private:

				static_assert(sizeof...(CM) > 0u, "The number of character class member must be positive.");

				/**
				 * @brief Match one character class member.
				 * @tparam M The type of class member.
				*/
				template<class M>
				struct MemberSpecification;

				//Given a character input, search if the character class contains this character.
#define DECLARE_CLASS_MEMBER_CONTAINS static bool contains(char) noexcept

				//specialisation for each character class type
				template<char C>
				struct MemberSpecification<STPCharacterClass::Atomic<C>> {
				public:

					DECLARE_CLASS_MEMBER_CONTAINS;
			
				};

				template<char First, char Last>
				struct MemberSpecification<STPCharacterClass::Range<First, Last>> {
				public:

					DECLARE_CLASS_MEMBER_CONTAINS;

				};

				template<class... C>
				struct MemberSpecification<STPCharacterClass::Except<C...>> {
				public:

					DECLARE_CLASS_MEMBER_CONTAINS;

				};

#undef DECLARE_CLASS_MEMBER_CONTAINS

			public:

				DECLARE_REGLANG_MATCHER;
		
			};

		}

		/**
		 * @brief STPQuantifier allows repeating a matching expression for some number of time.
		 * All quantifiers are lazy.
		*/
		namespace STPQuantifier {

			//Special value for matching unlimited number of maximum number of repetition.
			inline constexpr size_t Unlimited = std::numeric_limits<size_t>::max();

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

			//Match an expression between zero and one time.
			template<class Expr>
			using Maybe = Repeat<Expr, 0u, 1u>;
			//Match an expression between zero and unlimited times.
			template<class Expr>
			using MaybeMany = Repeat<Expr, 0u, STPQuantifier::Unlimited>;
			//Match an expression between one and unlimited times.
			template<class Expr>
			using StrictMany = Repeat<Expr, 1u, STPQuantifier::Unlimited>;

		}

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