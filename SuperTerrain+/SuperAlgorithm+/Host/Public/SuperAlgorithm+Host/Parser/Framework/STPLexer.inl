//TEMPLATE DEFINITION FOR THE GENERAL PURPOSE LEXER UTILITY
#ifdef _STP_LEXER_H_

//Exception
#include <SuperTerrain+/Exception/STPInvalidEnum.h>
#include <SuperTerrain+/Exception/STPParserError.h>

#include <iterator>
#include <sstream>
#include <algorithm>

#define LEXER_TEMPLATE template<class... LexState>
#define NAMESPACE_LEXER_NAME SuperTerrainPlus::STPAlgorithm::STPLexer<LexState...>

/* ------------------------------------------------------- Token ---------------------------------------------------------- */
LEXER_TEMPLATE
constexpr NAMESPACE_LEXER_NAME::STPToken::STPToken() noexcept : Name(&STPToken::Nameless), TokenID(STPLexical::NullTokenID) {

}

LEXER_TEMPLATE
constexpr NAMESPACE_LEXER_NAME::STPToken::STPToken(const std::string_view& name, const STPLexical::STPTokenID id, const std::string_view lexeme) noexcept :
	Name(&name), TokenID(id), Lexeme(lexeme) {

}

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPToken::operator bool() const noexcept {
	return this->TokenID != STPLexical::NullTokenID;
}

LEXER_TEMPLATE
inline const SuperTerrainPlus::STPAlgorithm::STPStringViewAdaptor&
	NAMESPACE_LEXER_NAME::STPToken::operator*() const noexcept {
	return this->Lexeme;
}

LEXER_TEMPLATE
inline const SuperTerrainPlus::STPAlgorithm::STPStringViewAdaptor*
	NAMESPACE_LEXER_NAME::STPToken::operator->() const noexcept {
	return &this->Lexeme;
}

LEXER_TEMPLATE
template<class TokExpr>
inline bool NAMESPACE_LEXER_NAME::STPToken::operator==(TokExpr) const noexcept {
	if constexpr (STPToken::template isEOS<TokExpr>) {
		return this->TokenID == STPLexical::EndOfSequenceTokenID;
	} else {
		return this->TokenID == TokExpr::LexicalTokenExpressionID;
	}
}

LEXER_TEMPLATE
template<class TokExpr>
inline bool NAMESPACE_LEXER_NAME::STPToken::operator!=(TokExpr) const noexcept {
	return !this->operator==(TokExpr {});
}

/* --------------------------------------------------- Lexical State -------------------------------------- */
LEXER_TEMPLATE
constexpr NAMESPACE_LEXER_NAME::STPLexicalState::STPLexicalState() noexcept :
	//get the state ID of the first state
	StateID(std::tuple_element<0u, STPLexer::LexicalStateCollection>::type::LexicalStateID), StateIndex(0u) {

}

LEXER_TEMPLATE
template<class... E>
constexpr size_t NAMESPACE_LEXER_NAME::STPLexicalState::toStateIndex(const STPLexical::STPStateID state, std::tuple<E...>) noexcept {
	//compile-time linear search for the index
	size_t index = std::numeric_limits<size_t>::max();
	//search using fold expression; make it volatile to prevent compiler from optimising it away
	//this expression should always be true
	[[maybe_unused]] const volatile bool isValidStateID = ((index = E::Index, state == E::ID) || ...);
	return index;
}

LEXER_TEMPLATE
constexpr size_t NAMESPACE_LEXER_NAME::STPLexicalState::toStateIndex(const STPLexical::STPStateID state) noexcept {
	return STPLexicalState::toStateIndex(state, STPLexicalState::LexicalStateCollectionIDMap {});
}

LEXER_TEMPLATE
inline typename NAMESPACE_LEXER_NAME::STPLexicalState& NAMESPACE_LEXER_NAME::STPLexicalState::operator=(const STPLexical::STPStateID id) noexcept {
	//update the state ID
	this->StateID = id;
	//update the index
	this->StateIndex = STPLexicalState::toStateIndex(id);

	return *this;
}

LEXER_TEMPLATE
template<class State>
inline typename NAMESPACE_LEXER_NAME::STPLexicalState& NAMESPACE_LEXER_NAME::STPLexicalState::operator=(State) noexcept {
	(*this) = State::LexicalStateID;
	return *this;
}

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexicalState::operator SuperTerrainPlus::STPAlgorithm::STPLexical::STPStateID() const noexcept {
	return this->StateID;
}

LEXER_TEMPLATE
inline size_t NAMESPACE_LEXER_NAME::STPLexicalState::index() const noexcept {
	return this->StateIndex;
}

/* --------------------------------------------------------------------------------------------------------- */

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexer(const std::string_view input, const std::string_view lexer_name,
	const std::string_view source_name, const STPLexical::STPBehaviour& behaviour) noexcept :
	Behaviour(behaviour), Sequence(input), Line(1u), Character(1u), LexerName(lexer_name), SourceName(source_name) {

}

LEXER_TEMPLATE
template<class TokExpr>
constexpr const std::string_view& NAMESPACE_LEXER_NAME::getTokenName() noexcept {
	if constexpr (STPToken::template isEOS<TokExpr>) {
		return STPToken::EOSName;
	} else {
		return TokExpr::Representation;
	}
}

LEXER_TEMPLATE
template<class... ExpTokExpr>
inline void NAMESPACE_LEXER_NAME::throwUnexpectedTokenError(const STPToken& got_token) const {
	constexpr static auto AllStateName = std::array { LexState::Representation... };
	//get current state name
	const std::string_view& stateName = AllStateName[this->LexicalState.index()];

	//compose error message
	using std::endl;
	std::ostringstream err;

	err << "An unexpected token has encountered!" << endl;
	err << "In lexical state <" << stateName << '>' << endl;
	err << "Was expecting:" << endl;
	if constexpr (sizeof...(ExpTokExpr) == 0u) {
		//no expected token
		err << "Any valid token" << endl;
	} else {
		((err << '<' << STPLexer::getTokenName<ExpTokExpr>() << "> "), ...) << endl;
	}
	err << "Got token \'" << *got_token.Name << "\' with token ID: " << got_token.TokenID
		<< ", and matched string: " << ((*got_token)->empty() ? "<empty matching>" : *got_token.Lexeme) << endl;

	this->throwSyntaxError(err.str(), "unexpected token");
}

LEXER_TEMPLATE
inline std::string_view NAMESPACE_LEXER_NAME::pop(const size_t count) {
	using std::string_view;
	//extract a number of characters from the sequence
	const string_view lexeme = this->Sequence.substr(0u, count);
	this->Sequence.remove_prefix(count);

	const auto [lineBreaker] = this->Behaviour;
	//line and character number correction
	//find the number of newline symbol from the matched sequence
	if (const size_t newlineCount = std::count(lexeme.cbegin(), lexeme.cend(), lineBreaker);
		newlineCount > 0u) {
		this->Line += newlineCount;
		//find the new character position in the current line, if newline exists
		this->Character = lexeme.length() - lexeme.find_last_of(lineBreaker);
	} else {
		this->Character += lexeme.length();
	}

	return lexeme;
}

LEXER_TEMPLATE
template<SuperTerrainPlus::STPAlgorithm::STPLexical::STPStateID S>
inline auto NAMESPACE_LEXER_NAME::nextAtState(const std::string_view sequence) noexcept {
	using std::tuple_element_t;
	//put all token expression data from all states into a tuple
	//this is a tuple of array, each tuple is a state, and each array element is a token expression in a state
	constexpr static auto ExpressionFromAllState =
		std::make_tuple(STPLexer::getTokenExpressionData(typename LexState::TokenExpressionCollection {})...);
	//get state information
	constexpr static size_t CurrentStateIndex = STPLexicalState::toStateIndex(S);
	using CurrentState = tuple_element_t<CurrentStateIndex, STPLexer::LexicalStateCollection>;
	using CurrentTokenExpression = typename CurrentState::TokenExpressionCollection;
	using CurrentTokenExpressionDataType = typename tuple_element_t<CurrentStateIndex, decltype(ExpressionFromAllState)>::value_type;

	//get matching result
	const auto matchResult = STPLexer::STPTokenExpressionUtility<CurrentTokenExpression>::match(sequence);

	using ReturnPair = std::pair<const STPRegularLanguage::STPMatchLength, const CurrentTokenExpressionDataType* const>;
	//run the token matching rule
	//we want to match the longest sequence
	//if there exists more than one longest token, take the one that is declared first
	const auto max_length_it = std::max_element(matchResult.cbegin(), matchResult.cend(), [](const auto& a, const auto& b) {
		//treat special length of null as an infinitely small value
		//this operator satisfies the *Compare* requirement
		if (!b) {
			//also if (a == b == Null)
			return false;
		}
		if (!a) {
			return true;
		}
		return *a < *b;
	});
	if (max_length_it == matchResult.cend() || !*max_length_it) {
		constexpr static ReturnPair NullReturn(std::nullopt, nullptr);
		//no valid matching is found
		return NullReturn;
	}
	//found a valid match, get the token expression information
	const size_t token_idx = static_cast<size_t>(std::distance(matchResult.cbegin(), max_length_it));
	return ReturnPair(*max_length_it, std::get<CurrentStateIndex>(ExpressionFromAllState).data() + token_idx);
}

LEXER_TEMPLATE
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::next() {
	//matchers for each state
	constexpr static auto AllExpressionMatcher = STPLexer::instantiateStateMatcher(STPLexer::LexicalStateCollectionID {});
	using std::string_view;

	//2 counters to record the lexeme prefix for different lexical actions
	size_t prefix_begin = 0u, remainder_begin = 0u;
	while (true) {
		const string_view remainderSeq = this->Sequence.substr(remainder_begin);
		//check if there is anything remaining to be matched in every loop cycle
		if (remainderSeq.empty()) {
			constexpr static STPToken EndOfSequenceToken = STPToken(STPLexer::getTokenName<STPLexical::EndOfSequence>(),
				STPLexical::EndOfSequenceTokenID, string_view());
			return EndOfSequenceToken;
		}

		//look for the current state matching function and match it on the remainder sequence (whole sequence with prefixed removed)
		const auto [match_length, token_data] = AllExpressionMatcher[this->LexicalState.index()](remainderSeq);
		if (!match_length) {
			constexpr static STPToken NullToken;
			//no valid match found
			return NullToken;
		}

		//a valid matching is found
		const auto& [token_string, token_id, token_action, token_next_state] = *token_data;
		//update lexical state, if necessary
		if (token_next_state != STPLexical::NullStateID) {
			this->LexicalState = token_next_state;
		}

		//update sequence length
		remainder_begin += *match_length;
		//perform lexical action
		//other than consume action, we need to keep looping through the matching process
		using LA = STPLexical::STPAction;
		switch (token_action) {
		case LA::Discard:
			//throw away all previously matched string, and the current matching; then keep going
			prefix_begin = remainder_begin;
			[[fallthrough]];
		case LA::Collect:
			//concatenate the matched string to the previous matching; then keep going
			//basically we don't need to do anything
			break;
		case LA::Consume:
			//consume all matched string (including non-discarded prefix) and create a new token
			{
				//the popped lexeme includes the sequence we have discarded and the lexeme we want
				string_view lexeme = this->pop(remainder_begin);
				//then we remove discarded sequence
				lexeme.remove_prefix(prefix_begin);
				return STPToken(token_string, token_id, lexeme);
			}
		default:
			throw STP_INVALID_ENUM_CREATE(token_action, STPLexical::STPAction);
		}
	}
}

LEXER_TEMPLATE
inline void NAMESPACE_LEXER_NAME::throwSyntaxError(const std::string& message, const char* const error_title) const {
	throw STP_PARSER_INVALID_SYNTAX_CREATE(message, this->LexerName.data(), error_title,
		(STPException::STPParserError::STPInvalidSyntax::STPSourceInformation {
			std::string(this->SourceName), this->Line, this->Character
		})
	);
}

LEXER_TEMPLATE
template<class... ExpTokExpr>
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::expect() {
	//fetch the next token
	const STPToken nextToken = this->next();
	if (!nextToken || ((nextToken != ExpTokExpr {}) && ... && (sizeof...(ExpTokExpr) > 0u))) {
		//no matched token
		this->throwUnexpectedTokenError<ExpTokExpr...>(nextToken);
	}
	return nextToken;
}

#undef NAMESPACE_LEXER_NAME
#undef LEXER_TEMPLATE

#endif//_STP_LEXER_H_