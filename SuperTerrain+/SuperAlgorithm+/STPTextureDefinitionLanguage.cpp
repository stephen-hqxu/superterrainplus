#include <SuperAlgorithm+/Parser/STPTextureDefinitionLanguage.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//Matching
#include <ctype.h>
#include <charconv>

#include <memory>
#include <optional>
#include <algorithm>

//Stream
#include <iostream>
#include <sstream>
#include <type_traits>

using SuperTerrainPlus::STPDiversity::Sample;
using SuperTerrainPlus::STPDiversity::STPTextureDatabase;
using namespace SuperTerrainPlus::STPAlgorithm;

using std::vector;
using std::string_view;
using std::stringstream;
using std::make_pair;
using std::make_optional;
using std::unique_ptr;
using std::make_unique;

using std::distance;

using std::ostream;
using std::endl;

class STPTextureDefinitionLanguage::STPTDLLexer {
public:

	/**
	 * @brief STPToken specifies a lexer token from an input.
	*/
	struct STPToken {
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
			RightArrow = '>',
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
			switch (type) {
			case STPType::Number: os << "Number";
				break;
			case STPType::String: os << "String";
				break;
			case STPType::End: os << "End of File";
				break;
			case STPType::Invalid: os << "Invalid Syntax";
				break;
			default:
				os << static_cast<std::underlying_type_t<STPType>>(type);
			}
			return os;
		}

		//The type of this token
		const STPType Type;
		//The lexeme of this token
		const string_view Lexeme;

		/**
		 * @brief Create a new token.
		 * @param type The type of this token.
		 * @param beg The beginning iterator of this token.
		 * @param count The number of character this token contains.
		*/
		STPToken(STPType type, const char* beg, size_t count = 1ull) : Type(type), Lexeme(beg, count) {

		}

		/**
		 * @brief Create a new token.
		 * @param type The type of this token.
		 * @param beg The beginning iterator of this token.
		 * @param end The end iterator of this token.
		*/
		STPToken(STPType type, const char* beg, const char* end) : Type(type), Lexeme(beg, distance(beg, end)) {

		}

		~STPToken() = default;

	};

private:

	const string_view Source;
	//The input parsing string sequence in a string stream
	const char* Sequence;
	size_t Line = 1ull;
	size_t Ch = 1ull;

	/**
	 * @brief Peek at the first character in the string sequence.
	 * @return The first character in the string sequence.
	*/
	inline char peek() {
		return *this->Sequence;
	}

	/**
	 * @brief Remove the first character from the string sequence.
	*/
	inline void pop() {
		this->Ch++;
		this->Sequence++;
	}

	/**
	 * @brief Create a single character token
	 * @param type The type of the token
	 * @return A single character token
	*/
	inline STPToken atom(STPToken::STPType type) {
		const char* character = this->Sequence;
		//we can safely move the pointer forward because the pointer in the character is not owned by the view
		this->pop();
		return STPToken(type, character);
	}

	/**
	 * @brief Read the whole string until a non-alphabet character is encountered.
	 * @return A complete string token
	*/
	STPToken readString() {
		const char* start = this->Sequence;

		//keep pushing pointer forward until we see something
		do {
			this->pop();
		} while (isalpha(this->peek()));

		return STPToken(STPToken::STPType::String, start, this->Sequence);
	}

	/**
	 * @brief Read the whole number.
	 * @return A complete string token of valid number
	*/
	STPToken readNumber() {
		const char* start = this->Sequence;

		char identifier;
		//we need to be able to identify floating point number
		//we don't need to worry about invalid numeric format right now, for example 1.34.6ff54
		do {
			this->pop();
		} while (identifier = this->peek(), (isdigit(identifier) || identifier == '.' || identifier == 'f' || identifier == 'u'));

		return STPToken(STPToken::STPType::Number, start, this->Sequence);
	}

	/**
	 * @brief Get the next token from the input
	 * @return The next token in the input sequence
	*/
	STPToken next() {
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
		case '\0': return STPToken(STPToken::STPType::End, this->Sequence);

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
		case '>': return this->atom(STPToken::STPType::RightArrow);
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
	STPTDLLexer(const string_view& source) : Source(source), Sequence(this->Source.data()) {

	}

	STPTDLLexer(const STPTDLLexer&) = delete;

	STPTDLLexer(STPTDLLexer&&) = delete;

	STPTDLLexer& operator=(const STPTDLLexer&) = delete;

	STPTDLLexer& operator=(STPTDLLexer&&) = delete;

	~STPTDLLexer() = default;

	/**
	 * @brief Compose initial error message about the parsing error, which contains line number and character location.
	 * @param str The pointer to the input string stream.
	 * @param error_type A string to represent the type of error.
	 * @return The same string stream.
	*/
	stringstream& composeInitialErrorInfo(stringstream& str, const char* error_type) const {
		str << "Texture Definition Language(" << this->Line << ',' << this->Ch << "): " << error_type << endl;
		return str;
	}

	/**
	 * @brief Expect the next token to be some types.
	 * @tparam Type The collection of types expected.
	 * @param expected_type The type to be expected.
	 * @return The token with any type matched.
	 * If the next token does not have the same type as any of the expected, exception will be thrown.
	*/
	template<class... Type>
	STPToken expect(Type... expected_type) {
		const STPToken nextToken = this->next();
		if (((nextToken.Type != expected_type) && ...)) {
			//throw errors to indicate unexpected token.
			stringstream msg;
			this->composeInitialErrorInfo(msg, "unexpected token") 
				<< "Was expecting: " << endl;
			((msg << '\'' << expected_type << "\' "), ...) << endl;
			msg << "Got: " << endl;
			msg << '\'' << nextToken.Lexeme << '\'' << endl;

			throw STPException::STPInvalidSyntax(msg.str().c_str());
		}
		return nextToken;
	}

	/**
	 * @brief Convert a view of string to a numeric value.
	 * @tparam T The type of the number.
	 * @param view The pointer to the view of the string.
	 * @return The converted string, or exception if the format is invalid.
	*/
	template<typename T>
	T fromStringView(const string_view& str) const {
		using std::errc;
		T result = static_cast<T>(0);
		const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);

		if (ec == errc()) {
			//OK
			return result;
		}

		//Error
		stringstream err_msg;
		//check error type
		switch (ec) {
		case errc::invalid_argument:
			this->composeInitialErrorInfo(err_msg, "invalid number")
				<< "Numeric value \'" << str << "\' is not an identifiable valid number." << endl;
			break;
		case errc::result_out_of_range:
			this->composeInitialErrorInfo(err_msg, "numeric overflow")
				<< "The specified number \'" << str << "\' is too large and will overflow." << endl;
			break;
		default:
			//impossible
			break;
		}
		throw STPException::STPInvalidSyntax(err_msg.str().c_str());
	}

};

STPTextureDefinitionLanguage::STPTextureDefinitionLanguage(const string_view& source) {
	typedef STPTDLLexer::STPToken::STPType TokenType;
	STPTDLLexer lexer(source);

	//start doing lexical analysis and parsing
	while (true) {
		//check while identifier is it
		if (lexer.expect(TokenType::Hash, TokenType::End).Type == TokenType::End) {
			//end of file
			break;
		}
		const string_view operation = lexer.expect(TokenType::String).Lexeme;

		//depends on operations, we process them differently
		if (operation == "texture") {
			this->processTexture(lexer);
		}
		else if (operation == "rule") {
			this->processRule(lexer);
		}
		else if (operation == "group") {
			this->processGroup(lexer);
		}
		else {
			//invalid operation
			stringstream msg;
			lexer.composeInitialErrorInfo(msg, "unknown operation")
				<< "Operation code \'" << operation << "\' is undefined by Texture Definition Language." << endl;
			throw STPException::STPInvalidSyntax(msg.str().c_str());
		}
	}
	
}

STPTextureDefinitionLanguage::~STPTextureDefinitionLanguage() = default;

void STPTextureDefinitionLanguage::checkTextureDeclared(const STPTDLLexer& lexer, const string_view& texture) const {
	if (this->DeclaredTexture.find(texture) == this->DeclaredTexture.cend()) {
		//texture variable not found, throw error
		stringstream msg;
		lexer.composeInitialErrorInfo(msg, "unknown texture")
			<< "Texture \'" << texture << "\' is not declared before it is being referenced." << endl;
		throw STPException::STPInvalidSyntax(msg.str().c_str());
	}
}

void STPTextureDefinitionLanguage::processTexture(STPTDLLexer& lexer) {
	typedef STPTDLLexer::STPToken::STPType TokenType;
	//declare some texture variables for texture ID
	lexer.expect(TokenType::LeftSquare);

	while (true) {
		const string_view textureName = lexer.expect(TokenType::String).Lexeme;
		//found a texture, store it
		//initially the texture has no view information
		this->DeclaredTexture.emplace(textureName, STPTextureDefinitionLanguage::UnreferencedIndex);

		if (lexer.expect(TokenType::Comma, TokenType::RightSquare).Type == TokenType::RightSquare) {
			//no more texture
			break;
		}
		//a comma means more texture are coming...
	}

	lexer.expect(TokenType::Semicolon);
}

void STPTextureDefinitionLanguage::processRule(STPTDLLexer& lexer) {
	typedef STPTDLLexer::STPToken::STPType TokenType;

	//define a rule
	const string_view rule_type = lexer.expect(TokenType::String).Lexeme;
	lexer.expect(TokenType::LeftCurly);

	while (true) {
		//we got a sample ID
		const Sample rule4Sample = lexer.fromStringView<Sample>(lexer.expect(TokenType::Number).Lexeme);
		lexer.expect(TokenType::Colon);
		lexer.expect(TokenType::Equal);
		lexer.expect(TokenType::LeftBracket);

		//start parsing rule
		while (true) {

			//check which type of type we are parsing
			if (rule_type == "altitude") {
				const float altitude = lexer.fromStringView<float>(lexer.expect(TokenType::Number).Lexeme);
				lexer.expect(TokenType::Minus);
				lexer.expect(TokenType::RightArrow);
				const string_view map2Texture = lexer.expect(TokenType::String).Lexeme;
				this->checkTextureDeclared(lexer, map2Texture);

				//store an altitude rule
				this->Altitude.emplace_back(rule4Sample, altitude, map2Texture);
			}
			else if (rule_type == "gradient") {
				const float minG = lexer.fromStringView<float>(lexer.expect(TokenType::Number).Lexeme);
				lexer.expect(TokenType::Comma);
				const float maxG = lexer.fromStringView<float>(lexer.expect(TokenType::Number).Lexeme);
				lexer.expect(TokenType::Comma);
				const float LB = lexer.fromStringView<float>(lexer.expect(TokenType::Number).Lexeme);
				lexer.expect(TokenType::Comma);
				const float UB = lexer.fromStringView<float>(lexer.expect(TokenType::Number).Lexeme);
				lexer.expect(TokenType::Minus);
				lexer.expect(TokenType::RightArrow);
				const string_view map2Texture = lexer.expect(TokenType::String).Lexeme;
				this->checkTextureDeclared(lexer, map2Texture);

				//store a gradient rule
				this->Gradient.emplace_back(rule4Sample, minG, maxG, LB, UB, map2Texture);
			}
			else {
				stringstream msg;
				lexer.composeInitialErrorInfo(msg, "unrecognised rule type")
					<< "Rule type \'" << rule_type << "\' is not recognised." << endl;
				throw STPException::STPInvalidSyntax(msg.str().c_str());
			}

			if (lexer.expect(TokenType::Comma, TokenType::RightBracket).Type == TokenType::RightBracket) {
				//no more rule setting
				break;
			}
		}

		if (lexer.expect(TokenType::Comma, TokenType::RightCurly).Type == TokenType::RightCurly) {
			//no more rule
			break;
		}
	}
}

void STPTextureDefinitionLanguage::processGroup(STPTDLLexer& lexer) {
	typedef STPTDLLexer::STPToken::STPType TokenType;

	//the declared type of this group
	const string_view group_type = lexer.expect(TokenType::String).Lexeme;
	lexer.expect(TokenType::LeftCurly);
	
	//record all textures to be added to a new group,
	//always make sure data being parsed are valid before adding to the parsing memory.
	vector<string_view> texture_in_group;
	//for each group
	while (true) {
		//clear old data
		texture_in_group.clear();

		//for each texture name assigned to this group
		while (true) {
			const string_view& assignedTexture = texture_in_group.emplace_back(lexer.expect(TokenType::String).Lexeme);
			this->checkTextureDeclared(lexer, assignedTexture);

			if (lexer.expect(TokenType::Comma, TokenType::Colon).Type == TokenType::Colon) {
				//no more texture to be assigned to this group
				break;
			}
		}
		lexer.expect(TokenType::Equal);

		//beginning of a group definition tuple
		lexer.expect(TokenType::LeftBracket);
		if (group_type == "view") {
			STPTextureDatabase::STPViewGroupDescription& view = this->DeclaredViewGroup.emplace_back();

			view.PrimaryScale = lexer.fromStringView<unsigned int>(lexer.expect(TokenType::Number).Lexeme);
			lexer.expect(TokenType::Comma);
			view.SecondaryScale = lexer.fromStringView<unsigned int>(lexer.expect(TokenType::Number).Lexeme);
			lexer.expect(TokenType::Comma);
			view.TertiaryScale = lexer.fromStringView<unsigned int>(lexer.expect(TokenType::Number).Lexeme);
		}
		else {
			stringstream msg;
			lexer.composeInitialErrorInfo(msg, "unrecognised group type")
				<< "Group type \'" << group_type << "\' is not recognised." << endl;
			throw STPException::STPInvalidSyntax(msg.str().c_str());
		}
		//end of a group definition tuple
		lexer.expect(TokenType::RightBracket);

		//assign texture with group index
		std::for_each(texture_in_group.cbegin(), texture_in_group.cend(), 
		[&texture_table = this->DeclaredTexture, view_group_index = this->DeclaredViewGroup.size() - 1ull](const auto& name) {
			//assign all texture to be added with the group just parsed
			texture_table[name] = view_group_index;
		});

		if (lexer.expect(TokenType::Comma, TokenType::RightCurly).Type == TokenType::RightCurly) {
			//end of group
			break;
		}
	}
}

STPTextureDefinitionLanguage::STPTextureVariable STPTextureDefinitionLanguage::operator()(STPTextureDatabase& database) const {
	//prepare variable dictionary for return
	STPTextureVariable varDic;
	const unsigned int textureCount = static_cast<unsigned int>(this->DeclaredTexture.size());
	varDic.reserve(textureCount);

	namespace TI = SuperTerrainPlus::STPDiversity::STPTextureInformation;
	//to convert view group index to corresponded ID in the database
	unique_ptr<TI::STPViewGroupID[]> ViewGroupIDLookup = make_unique<TI::STPViewGroupID[]>(this->DeclaredViewGroup.size());
	//add texture view group
	std::transform(this->DeclaredViewGroup.cbegin(), this->DeclaredViewGroup.cend(), ViewGroupIDLookup.get(), [&database](const auto& view_desc) {
		return database.addViewGroup(view_desc);
	});

	//requesting texture
	//assign each variable with those texture ID
	for (auto [texture_it, i] = make_pair(this->DeclaredTexture.cbegin(), 0u); texture_it != this->DeclaredTexture.cend(); texture_it++, i++) {
		const auto& [texture_name, view_group_index] = *texture_it;
		if (view_group_index == STPTextureDefinitionLanguage::UnreferencedIndex) {
			//this index has no corresponding view group
			stringstream msg;
			msg << "View group reference for \'" << texture_name << "\' is undefined." << endl;

			throw STPException::STPMemoryError(msg.str().c_str());
		}

		const TI::STPViewGroupID view_group_id = ViewGroupIDLookup[view_group_index];
		varDic.try_emplace(texture_name, database.addTexture(view_group_id, make_optional(texture_name)), view_group_id);
	}

	//add splat rules into the database
	//when we were parsing the TDL, we have already checked all used texture are declared, so we are sure textureName can be located in the dictionary
	STPTextureDatabase::STPTextureSplatBuilder& splat_builder = database.getSplatBuilder();
	for (const auto& [sample, ub, textureName] : this->Altitude) {
		//one way to call function using tuple is std::apply, however we need to replace textureName with textureID in this database.
		splat_builder.addAltitude(sample, ub, varDic[textureName].first);
	}
	for (const auto& [sample, minG, maxG, lb, ub, textureName] : this->Gradient) {
		splat_builder.addGradient(sample, minG, maxG, lb, ub, varDic[textureName].first);
	}

	//DONE!!!
	return varDic;
}