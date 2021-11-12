#include <SuperTerrain+/World/Diversity/Texture/STPTextureDefinitionLanguage.h>

//Error
#include <SuperTerrain+/Utility/Exception/STPInvalidSyntax.h>

//Matching
#include <ctype.h>

#include <algorithm>

//Stream
#include <iostream>
#include <sstream>
#include <type_traits>

using namespace SuperTerrainPlus::STPDiversity;

using std::string;
using std::string_view;
using std::make_unique;

using std::distance;

using std::ostream;

class STPTextureDefinitionLanguage::STPTDLLexer {
public:

	/**
	 * @brief STPToken specifies a lexer token from an input.
	*/
	class STPToken {
	public:

		/**
		 * @brief STPType denotes the type of this token
		*/
		enum class STPType : char {
			//0-9 numeric value
			Number = 0x00,
			//a-z and A-Z ASCII char string
			String = 0x01,
			//A [ symbol
			LeftSquare = '[',
			//A ] symbol
			RightSquare = ']',
			//A { symbol
			LeftCurly = '{',
			//A } symbol
			RightCurly = '}',
			//A ( symbol
			LeftBracket = '(',
			//A ) symbol
			RightBracket = ')',
			//A # symbol
			Hash = '#',
			//A , symbol
			Comma = ',',
			//A ; symbol
			Semicolon = ';',
			//A - symbol
			Minus = '-',
			//A > symbol
			Right = '>',
			//A : symbol
			Colon = ':',
			//A = symbol
			Equal = '=',
			//The deliminator for end of string
			End = 0x02,
			//Not a valid token specified by TDL
			Invalid = 0x03
		};

		friend ostream& operator<<(ostream& os, STPType type) {
			//Special type names for some of the types, can use enum value as index
			static constexpr string_view SpecialTypeName[4] = {
				"Number",
				"String",
				"End of File",
				"Invalid Syntax"
			};
			using Type = std::underlying_type_t<STPToken::STPType>;
			const Type typeValue = static_cast<Type>(type);

			//some of the type contains a char array rather than a char, we want to replace them with more detailed name.
			if (typeValue < 4) {
				os << SpecialTypeName[typeValue];
			}
			else {
				os << typeValue;
			}
			return os;
		}

	private:

		//The type of this token
		STPType Type;
		//The lexeme of this token
		string_view Lexeme;

	public:

		/**
		 * @brief Create a new token.
		 * @param type The type of this token.
		 * @param beg The beginning iterator of this token.
		 * @param count The numebr of character this token contains.
		*/
		constexpr STPToken(STPType type, const char* beg, size_t count = 1ull) : Type(type), Lexeme(beg, count) {

		}

		/**
		 * @brief Create a new token.
		 * @param type The type of this token.
		 * @param beg The beginning interator of this token.
		 * @param end The end iterator of this token.
		*/
		constexpr STPToken(STPType type, const char* beg, const char* end) : Type(type), Lexeme(beg, distance(beg, end)) {

		}

		~STPToken() = default;

		/**
		 * @brief Get the token type.
		 * @return The type of this token
		*/
		inline STPType getType() const {
			return this->Type;
		}

		/**
		 * @brief Get the token lexeme.
		 * @return The lexeme of this token.
		*/
		inline const string_view& getLexeme() const {
			return this->Lexeme;
		}

	};

private:

	const string Source;
	//The input parsing string sequence in a string stream
	mutable string_view Sequence;
	mutable size_t Line = 1ull;
	mutable size_t Ch = 1ull;

	/**
	 * @brief Peek at the first character in the string sequence.
	 * @return The first character in the string sequence.
	*/
	constexpr char peek() const {
		return this->Sequence.front();
	}

	/**
	 * @brief Remove the first character from the string sequence.
	*/
	constexpr void pop() const {
		this->Ch++;
		this->Sequence.remove_prefix(1ull);
	}

	/**
	 * @brief Create a single character token
	 * @param type The type of the token
	 * @return A single character token
	*/
	constexpr STPToken atom(STPToken::STPType type) const {
		const char* character = this->Sequence.data();
		//we can safely move the pointer forward because the pointer in the character is not owned by the view
		this->pop();
		return STPToken(type, character);
	}

	/**
	 * @brief Read the whole string until a non-alphabet character is encountered.
	 * @return A complete string token
	*/
	STPToken readString() const {
		const char* start = this->Sequence.data();
		this->pop();

		//keep pushing pointer forward until we see something
		while (isalpha(this->peek())) {
			this->pop();
		}
		return STPToken(STPToken::STPType::String, start, this->Sequence.data());
	}

	/**
	 * @brief Read the whole number.
	 * @return A complete string token of valid number
	*/
	STPToken readNumber() const {
		const char* start = this->Sequence.data();
		this->pop();

		char identifier;
		//we need to be able to identify floating point number
		//we don't need to worry about invalid numeric format right now, for example 1.34.6ff54
		while (identifier = this->peek(), (isdigit(identifier) || identifier == '.' || identifier == 'f' || identifier == 'u')) {
			this->pop();
			identifier = this->peek();
		}
		return STPToken(STPToken::STPType::Number, start, this->Sequence.data());
	}

	/**
	 * @brief Get the next token from the input
	 * @return The next token in the input sequence
	*/
	inline STPToken next() const {
		{
			char space;
			while (space = this->peek(), isspace(space)) {
				if (space == '\n') {
					//record newline
					this->Line++;
					this->Ch = 1ull;
				}
				//white space and newline and tab characters are all ignored
				this->pop();
			}
		}
		const char identifier = this->peek();

		//checking for single character identifier
		switch (identifier) {
			//End of parser input
		case '\0': return STPToken(STPToken::STPType::End, this->Sequence.data());

		case '[': return this->atom(STPToken::STPType::LeftSquare);
		case ']': return this->atom(STPToken::STPType::RightSquare);
		case '{': return this->atom(STPToken::STPType::LeftCurly);
		case '}': return this->atom(STPToken::STPType::RightCurly);
		case '(': return this->atom(STPToken::STPType::LeftBracket);
		case ')': return this->atom(STPToken::STPType::RightBracket);

		case '#': return this->atom(STPToken::STPType::Hash);
		case ',': return this->atom(STPToken::STPType::Comma);
		case ';': return this->atom(STPToken::STPType::Semicolon);
		case '-': return this->atom(STPToken::STPType::Minus);
		case '>': return this->atom(STPToken::STPType::Right);
		case ':': return this->atom(STPToken::STPType::Colon);
		case '=': return this->atom(STPToken::STPType::Equal);
		default:
			//none of any single character identifier
			if (isalpha(identifier)) {
				//attempt to read the whole string
				return this->readString();
			}
			if (isdigit(identifier)) {
				//attempt to read the whole number
				return this->readNumber();
			}
			//none of the above? must be a syntax error
			return this->atom(STPToken::STPType::Invalid);
		}
	}

public:

	/**
	 * @brief Init the TDL lexer with source code
	 * @param source The source code to TDL
	*/
	STPTDLLexer(const string& source) : Source(source), Sequence(this->Source.data()) {

	}

	STPTDLLexer(const STPTDLLexer&) = delete;

	STPTDLLexer(STPTDLLexer&&) = delete;

	STPTDLLexer& operator=(const STPTDLLexer&) = delete;

	STPTDLLexer& operator=(STPTDLLexer&&) = delete;

	~STPTDLLexer() = default;

	/**
	 * @brief Expect the next token to be some types.
	 * @tparam Type The collection of types expected.
	 * @param expected_type The type to be expected.
	 * @return The token with any type matched.
	 * If the next token does not have the same type as any of the expected, exception will be thrown.
	*/
	template<class... Type>
	STPToken expect(Type... expected_type) const {
		const STPToken nextToken = this->next();
		if (((nextToken.getType() != expected_type) && ...)) {
			using std::endl;
			std::stringstream msg;
			msg << "Texture Deinition Language(" << this->Line << ',' << this->Ch << "): error" << endl;
			msg << "Was expecting: " << endl;
			((msg << '\'' << expected_type << "\' "), ...) << endl;
			msg << "Got: " << endl;
			msg << '\'' << nextToken.getLexeme() << '\'' << endl;

			throw STPException::STPInvalidSyntax(msg.str().c_str());
		}
		return nextToken;
	}

};

STPTextureDefinitionLanguage::STPTextureDefinitionLanguage(const string& source) : Lexer(make_unique<STPTDLLexer>(source)) {
	typedef STPTDLLexer::STPToken Token;
	typedef Token::STPType TokenType;
	auto stoSample = [](const string_view& str) -> Sample {
		//one disadvantage of this method is it will create a string from the string_view
		return static_cast<Sample>(std::stoull(str.data()));
	};
	auto stoFloat = [](const string_view& str) -> float {
		return std::stof(str.data());
	};

	//start doing lexical analysis and parsing
	while (true) {
		//check while identifier is it
		if (this->Lexer->expect(TokenType::Hash, TokenType::End).getType() == TokenType::End) {
			//end of file
			break;
		}
		const string_view operation = this->Lexer->expect(TokenType::String).getLexeme();

		if (operation == "texture") {
			//declare some texture variables for texture ID
			this->Lexer->expect(TokenType::LeftSquare);

			while (true) {
				const string_view textureName = this->Lexer->expect(TokenType::String).getLexeme();
				//a texture variable, store it
				this->DeclaredTextureVariable.emplace_back(textureName);

				if (this->Lexer->expect(TokenType::Comma, TokenType::RightSquare).getType() == TokenType::RightSquare) {
					//no more texture
					break;
				}
				//a comma means more texture are coming...
			}

			this->Lexer->expect(TokenType::Semicolon);
		}
		else if (operation == "rule") {
			//define a rule
			const string_view rule_type = this->Lexer->expect(TokenType::String).getLexeme();
			this->Lexer->expect(TokenType::LeftCurly);

			while (true) {
				//we got a sample ID
				const Sample rule4Sample = stoSample(this->Lexer->expect(TokenType::Number).getLexeme());
				this->Lexer->expect(TokenType::Colon);
				this->Lexer->expect(TokenType::Equal);
				this->Lexer->expect(TokenType::LeftBracket);

				//start parsing rule
				while (true) {

					//check which type of type we are parsing
					if (rule_type == "altitude") {
						const float altitude = stoFloat(this->Lexer->expect(TokenType::Number).getLexeme());
						this->Lexer->expect(TokenType::Minus);
						this->Lexer->expect(TokenType::Right);
						const string_view map2Texture = this->Lexer->expect(TokenType::String).getLexeme();

						//store an altitude rule
						this->Altitude.emplace_back(rule4Sample, altitude, map2Texture);
					}
					else if (rule_type == "gradient") {
						const float minG = stoFloat(this->Lexer->expect(TokenType::Number).getLexeme());
						this->Lexer->expect(TokenType::Comma);
						const float maxG = stoFloat(this->Lexer->expect(TokenType::Number).getLexeme());
						this->Lexer->expect(TokenType::Comma);
						const float LB = stoFloat(this->Lexer->expect(TokenType::Number).getLexeme());
						this->Lexer->expect(TokenType::Comma);
						const float UB = stoFloat(this->Lexer->expect(TokenType::Number).getLexeme());
						this->Lexer->expect(TokenType::Minus);
						this->Lexer->expect(TokenType::Right);
						const string_view map2Texture = this->Lexer->expect(TokenType::String).getLexeme();
						
						//store a gradient rule
						this->Gradient.emplace_back(rule4Sample, minG, maxG, LB, UB, map2Texture);
					}

					if (this->Lexer->expect(TokenType::Comma, TokenType::RightBracket).getType() == TokenType::RightBracket) {
						//no more rule setting
						break;
					}
				}

				if (this->Lexer->expect(TokenType::Comma, TokenType::RightCurly).getType() == TokenType::RightCurly) {
					//no more rule
					break;
				}
			}
		}
	}
	
}

STPTextureDefinitionLanguage::~STPTextureDefinitionLanguage() = default;