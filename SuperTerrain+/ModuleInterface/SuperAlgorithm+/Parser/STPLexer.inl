//TEMPLATE DEFINITION FOR THE GENERAL PURPOSE LEXER UTILITY
#ifdef _STP_LEXER_H_

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

#include <cctype>

#include <sstream>
#include <iterator>
#include <algorithm>

#define NAMESPACE_LEXER_NAME SuperTerrainPlus::STPAlgorithm::LEXER_NAME

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexer::STPToken::STPToken() noexcept : TokenID(STPLexerToken::NullTokenID), Lexeme("InvalidToken") {

}

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexer::STPToken::STPToken(const STPLexerToken::STPTokenID id, const std::string_view lexeme) noexcept :
	TokenID(id), Lexeme(lexeme) {

}

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexer::STPToken::operator bool() const noexcept {
	return this->TokenID != STPLexerToken::NullTokenID;
}

LEXER_TEMPLATE
inline const SuperTerrainPlus::STPAlgorithm::STPStringViewAdaptor&
	NAMESPACE_LEXER_NAME::STPLexer::STPToken::operator*() const noexcept {
	return this->Lexeme;
}

LEXER_TEMPLATE
inline const SuperTerrainPlus::STPAlgorithm::STPStringViewAdaptor*
	NAMESPACE_LEXER_NAME::STPLexer::STPToken::operator->() const noexcept {
	return &this->Lexeme;
}

LEXER_TEMPLATE
template<class Tok>
inline bool NAMESPACE_LEXER_NAME::STPLexer::STPToken::operator==(Tok&&) const noexcept {
	return this->TokenID == Tok::ID;
}

LEXER_TEMPLATE
template<class Tok>
inline bool NAMESPACE_LEXER_NAME::STPLexer::STPToken::operator!=(Tok&&) const noexcept {
	return this->TokenID != Tok::ID;
}

LEXER_TEMPLATE
inline NAMESPACE_LEXER_NAME::STPLexer(const char* const input, const std::string_view& lexer_name, const std::string_view& source_name) noexcept :
	Sequence(input), Line(1u), Character(1u), LexerName(lexer_name), SourceName(source_name) {

}

LEXER_TEMPLATE
template<class... ExpTok>
inline void NAMESPACE_LEXER_NAME::throwUnexpectedTokenError(const STPToken& got_token) const {
	//compose error message
	using std::endl;
	std::ostringstream err;

	err << "An unexpected token has encountered!" << endl;
	err << "Was expecting:" << endl;
	((err << ExpTok::Representation << ' '), ...) << endl;
	err << "Got token with token ID: " << got_token.TokenID << ", and value: " << *got_token.Lexeme << endl;

	this->throwSyntaxError(err.str(), "unexpected token");
}

LEXER_TEMPLATE
inline char NAMESPACE_LEXER_NAME::peek() const noexcept {
	return *this->Sequence;
}

LEXER_TEMPLATE
inline void NAMESPACE_LEXER_NAME::pop(const size_t count) noexcept {
	this->Sequence += count;
	this->Character += count;
}

LEXER_TEMPLATE
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::createToken(const STPLexerToken::STPTokenID id, const size_t length) noexcept {
	using std::string_view;
	const char* const tokenSeq_begin = this->Sequence;
	//we can safely move the pointer forward because the pointer in the character is not owned by the view
	this->pop(length);
	return STPToken(id, string_view(tokenSeq_begin, length));
}

LEXER_TEMPLATE
inline void NAMESPACE_LEXER_NAME::correctStats(const STPToken& token) noexcept {
	using std::string_view;
	//line and character number correction
	const string_view& seq = *token.Lexeme;
	//find the number of newline symbol from the matched sequence
	if (const size_t newlineCount = std::count(seq.cbegin(), seq.cend(), '\n');
		newlineCount > 0u) {
		this->Line += newlineCount;
		//find the new character position in the current line, if newline exists
		this->Character = seq.length() - seq.find_last_of('\n');
	}
}

LEXER_TEMPLATE
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::nextAtomicToken() noexcept {
	STPLexerToken::STPTokenID nextID = STPLexerToken::NullTokenID;
	const char nextAtom = this->peek();

	//find the first token with the character matched
	if (((nextID = AtomToken::ID, nextAtom == AtomToken::Character) || ...)) {
		//found
		return this->createToken(nextID);
	}
	//not found, returns an invalid token
	return STPToken();
}

LEXER_TEMPLATE
template<class First, class... Rest>
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::nextFunctionToken() noexcept {
	//call the functor on the first function token to match
	//returns the length of the sequence
	const size_t seqLen = First()(this->Sequence);
	if (seqLen == 0u) {
		//no match is found
		if constexpr (sizeof...(Rest) == 0u) {
			//no more function token to be used, nothing is matched
			return STPToken();
		} else {
			//proceed to the next function token
			return this->nextFunctionToken<Rest...>();
		}
	}
	//found a valid matching
	const STPToken matched = this->createToken(First::ID, seqLen);
	//since this function is user-defined, we need to correct the stats if it contains newline symbol
	this->correctStats(matched);

	return matched;
}

LEXER_TEMPLATE
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::next() noexcept {
	//preprocessing
	{
		//skip white space
		char c;
		while (c = this->peek(), std::isspace(c)) {
			if (c == '\n') {
				//record newline
				this->Line++;
				this->Character = 1u;
			}
			//white space and newline and tab characters are all ignored
			this->pop();
		}
		//check for the null terminator in the string
		if (c == '\0') {
			using std::string_view;
			//create a null token without popping from the string, otherwise that's UB
			//create an empty string view, because string view doesn't include null in the string length anyway
			return STPToken(STPLexerToken::Null::ID, string_view());
		}
	}

	//find atomic token first
	if constexpr (STPLexer::AtomTokenCount > 0u) {
		if (const STPToken atomicToken = this->nextAtomicToken();
			atomicToken) {
			return atomicToken;
		}
	}
	//if not, find function token
	if constexpr (STPLexer::FuncTokenCount > 0u) {
		if (const STPToken funcToken = this->nextFunctionToken<FuncToken...>();
			funcToken) {
			return funcToken;
		}
	}
	//nothing can be matched
	return STPToken();
}

LEXER_TEMPLATE
inline void NAMESPACE_LEXER_NAME::throwSyntaxError(const std::string& message, const char* error_title) const {
	throw STP_PARSER_INVALID_SYNTAX_CREATE(message, this->LexerName.data(), error_title,
		(STPException::STPParserError::STPInvalidSyntax::STPSourceInformation {
			std::string(this->SourceName), this->Line, this->Character
		})
	);
}

LEXER_TEMPLATE
template<class... ExpTok>
inline typename NAMESPACE_LEXER_NAME::STPToken NAMESPACE_LEXER_NAME::expect() {
	//fetch the next token
	const STPToken nextToken = this->next();
	if (!nextToken || ((nextToken != ExpTok {}) && ... && (sizeof...(ExpTok) > 0u))) {
		//no matched token
		this->throwUnexpectedTokenError<ExpTok...>(nextToken);
	}
	return nextToken;
}

#undef NAMESPACE_LEXER_NAME

#endif//_STP_LEXER_H_