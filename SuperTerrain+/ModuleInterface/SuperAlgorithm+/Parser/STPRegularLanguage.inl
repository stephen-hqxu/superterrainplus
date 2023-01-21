//TEMPLATE DEFINITION FOR REGULAR LANGUAGE

#ifdef _STP_REGULAR_LANGUAGE_H_

#define NAMESPACE_REGLANG SuperTerrainPlus::STPAlgorithm::STPRegularLanguage

/* ----------------------------------------- implementation for internal functions ----------------------------------- */
template<class Expr>
inline bool NAMESPACE_REGLANG::STPDetail::matchExpression(std::string_view& sequence, size_t& total_length) noexcept {
	//call the matching function for this expression
	const size_t match_length = Expr::match(sequence);
	if (match_length == STPRegularLanguage::NoMatch) {
		//no valid matching found
		return false;
	}

	//found a valid matching
	total_length += match_length;
	sequence.remove_prefix(match_length);
	return true;
}

/* ------------------------------------------------------------------------------------------------------------------- */
#define DEFINE_REGLANG_MATCHER(OP) \
inline size_t NAMESPACE_REGLANG::OP::match(const std::string_view& sequence) noexcept
#define OP_COMMA ,

/* ----------------------------------- Character Class ----------------------------------------- */
#define DEFINE_CLASS_MEMBER_CONTAINS(TEMP_ARG, LE_OP) template<class... CM> template<TEMP_ARG> \
inline bool NAMESPACE_REGLANG::STPCharacterClass::Class<CM...>::MemberSpecification< \
	NAMESPACE_REGLANG::STPCharacterClass::LE_OP>::contains(const char right) noexcept

DEFINE_CLASS_MEMBER_CONTAINS(char C, Atomic<C>) {
	//simple equality check
	return right == C;
}

DEFINE_CLASS_MEMBER_CONTAINS(char First OP_COMMA char Last, Range<First OP_COMMA Last>) {
	//range check, remember the range is inclusive on both ends
	return right >= First && right <= Last;
}

DEFINE_CLASS_MEMBER_CONTAINS(class... C, Except<C...>) {
	//just negate the matching result
	return !(Class::MemberSpecification<C>::contains(right) || ...);
}

#undef DEFINE_CLASS_MEMBER_CONTAINS

template<class... CM>
DEFINE_REGLANG_MATCHER(STPCharacterClass::Class<CM...>) {
	//sanity check
	if (sequence.empty()) {
		return STPRegularLanguage::NoMatch;
	}
	const char c = sequence.front();
	//check this character in the class of characters
	return (Class::MemberSpecification<CM>::contains(c) || ...) ? 1u : STPRegularLanguage::NoMatch;
}

/* -------------------------------------------- Quantifier --------------------------------------- */
template<class Expr, size_t Min, size_t Max>
DEFINE_REGLANG_MATCHER(STPQuantifier::Repeat<Expr OP_COMMA Min OP_COMMA Max>) {
	//create a copy for manipulation during the repeat matching
	std::string_view remainingSeq(sequence);
	size_t totalLength = 0u;

	//record the number of repetition
	size_t num_rep = 0u;
	for (; num_rep < Max; num_rep++) {
		if (!STPDetail::matchExpression<Expr>(remainingSeq, totalLength)) {
			//quit matching if expression fails on the remaining sequence
			break;
		}
	}

	//check if the number of repetition satisfies the requirement
	return num_rep >= Min ? totalLength : STPRegularLanguage::NoMatch;
}

/* ------------------------------------------------------------------------------------------------ */

DEFINE_REGLANG_MATCHER(Any) {
	//as long as the input is not empty, always match a character.
	return sequence.empty() ? STPRegularLanguage::NoMatch : 1u;
}

template<const std::string_view& L>
DEFINE_REGLANG_MATCHER(Literal<L>) {
	//TODO: use starts_with() in C++ 20

	//find() is too slow, use substr() and string compare; see their complexity on the specification
	//basically we want to match from the start of the input if it equals the literal
	//consider it as match if the prefix substring from the input is exactly the same as the literal
	return sequence.substr(0u, Literal::LiteralLength) == L ? Literal::LiteralLength : STPRegularLanguage::NoMatch;
}

template<class... Expr>
DEFINE_REGLANG_MATCHER(Alternative<Expr...>) {
	std::string_view currentSeq;
	size_t matchLength = 0u;

	//need to restore the state back to the original for ever sub-expression
	if (!((currentSeq = sequence, matchLength = 0u, STPDetail::matchExpression<Expr>(currentSeq, matchLength)) || ...)) {
		//none of the expression is successful?
		return STPRegularLanguage::NoMatch;
	}
	return matchLength;
}

template<class... Expr>
DEFINE_REGLANG_MATCHER(Sequence<Expr...>) {
	//we create a copy of the input, so if the sequence operator fail we can backtrack with the original copy.
	std::string_view remainingSeq(sequence);
	size_t totalLength = 0u;

	//run through all expressions
	if (!(STPDetail::matchExpression<Expr>(remainingSeq, totalLength) && ...)) {
		//a valid match of the sequence says all sub-expression have match, if not the sequence has no match.
		return STPRegularLanguage::NoMatch;
	}
	return totalLength;
}

#undef OP_COMMA
#undef DEFINE_REGLANG_MATCHER
#undef NAMESPACE_REGLANG

#endif//_STP_REGULAR_LANGUAGE_H_