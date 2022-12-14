#include <SuperAlgorithm+/Parser/STPTextureDefinitionLanguage.h>
#include <SuperAlgorithm+/Parser/STPLexer.h>

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

#include <cctype>
#include <sstream>

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

using std::tuple;
using std::string_view;
using std::ostringstream;
using std::vector;

using std::unique_ptr;
using std::make_unique;
using std::make_optional;
using std::as_const;

using std::for_each;

using std::endl;

using SuperTerrainPlus::STPDiversity::Sample;
using SuperTerrainPlus::STPDiversity::STPTextureDatabase;

using namespace SuperTerrainPlus::STPAlgorithm;

namespace LT = STPLexerToken;

//the name of our TDL parser
constexpr static char TDLParserName[] = "SuperTerrain+ Texture Definition Language";

//define our TDL lexer
namespace {
	//a string identifier in TDL
	STP_LEXER_DECLARE_FUNCTION_TOKEN(TDLIdentifier, 1000u, "TDL Identifier");
	//a numeric value in TDL, can be integer or decimal, and contains literal suffix
	STP_LEXER_DECLARE_FUNCTION_TOKEN(TDLNumber, 1001u, "TDL Number");

	typedef tuple<LT::LeftSquare, LT::RightSquare,
		LT::LeftCurly, LT::RightCurly,
		LT::LeftRound, LT::RightRound,
		LT::Hash, LT::Comma, LT::Minus, LT::GreaterThan, LT::Colon, LT::Equal> STPTDLAtomicToken;

	typedef tuple<TDLIdentifier, TDLNumber> STPTDLFunctionToken;

	//the lexer for TDL
	typedef STPLexer<STPTDLAtomicToken, STPTDLFunctionToken> STPTDLLexer;
}

inline STP_LEXER_DEFINE_FUNCTION_TOKEN(TDLIdentifier) {
	size_t strPos = 0u;
	//keep pushing pointer forward until we see a non-alphabet
	while (std::isalpha(sequence[strPos])) {
		strPos++;
	}
	return strPos;
}

inline STP_LEXER_DEFINE_FUNCTION_TOKEN(TDLNumber) {
	size_t numPos = 0u;
	//we need to be able to identify floating point number
	//we don't need to worry about invalid numeric format right now, for example 1.34.6ff54
	//invalid number format will be captured by the string parser later
	char c;
	while (c = sequence[numPos], (std::isdigit(c) || c == '.' || c == 'f' || c == 'u')) {
		numPos++;
	}
	return numPos;
}

using STPTextureDefinitionLanguage::STPResult;

static void checkTextureDeclared(const STPTDLLexer& lexer, STPResult& result, const string_view& texture) {
	if (result.DeclaredTexture.find(texture) == result.DeclaredTexture.cend()) {
		//texture variable not found, throw error
		ostringstream msg;
		msg << "Texture \'" << texture << "\' is not declared before it is being referenced." << endl;
		lexer.throwSyntaxError(msg.str(), "unknown texture");
	}
}

static void processTexture(STPTDLLexer& lexer, STPResult& result) {
	//declare some texture variables for texture ID
	lexer.expect<LT::LeftSquare>();

	while (true) {
		const string_view textureName = **lexer.expect<TDLIdentifier>();
		//found a texture, store it
		//initially the texture has no view information
		result.DeclaredTexture.emplace(textureName, STPResult::UnreferencedIndex);

		if (lexer.expect<LT::Comma, LT::RightSquare>() == LT::RightSquare {}) {
			//no more texture
			break;
		}
		//a comma means more texture are coming...
	}
}

static void processRule(STPTDLLexer& lexer, STPResult& result) {
	//define a rule
	const string_view rule_type = **lexer.expect<TDLIdentifier>();
	lexer.expect<LT::LeftCurly>();

	while (true) {
		//we got a sample ID
		const Sample rule4Sample = lexer.expect<TDLNumber>()->to<Sample>();
		lexer.expect<LT::Colon>();
		lexer.expect<LT::Equal>();
		lexer.expect<LT::LeftRound>();

		//start parsing rule
		while (true) {
			//check which type of type we are parsing
			if (rule_type == "altitude") {
				const float altitude = lexer.expect<TDLNumber>()->to<float>();
				lexer.expect<LT::Minus>();
				lexer.expect<LT::GreaterThan>();
				const string_view map2Texture = **lexer.expect<TDLIdentifier>();
				checkTextureDeclared(lexer, result, map2Texture);

				//store an altitude rule
				result.Altitude.emplace_back(rule4Sample, altitude, map2Texture);
			} else if (rule_type == "gradient") {
				const float minG = lexer.expect<TDLNumber>()->to<float>();
				lexer.expect<LT::Comma>();
				const float maxG = lexer.expect<TDLNumber>()->to<float>();
				lexer.expect<LT::Comma>();
				const float LB = lexer.expect<TDLNumber>()->to<float>();
				lexer.expect<LT::Comma>();
				const float UB = lexer.expect<TDLNumber>()->to<float>();
				lexer.expect<LT::Minus>();
				lexer.expect<LT::GreaterThan>();
				const string_view map2Texture = **lexer.expect<TDLIdentifier>();
				checkTextureDeclared(lexer, result, map2Texture);

				//store a gradient rule
				result.Gradient.emplace_back(rule4Sample, minG, maxG, LB, UB, map2Texture);
			} else {
				ostringstream msg;
				msg << "Rule type \'" << rule_type << "\' is not recognised." << endl;
				lexer.throwSyntaxError(msg.str(), "unrecognised rule type");
			}

			if (lexer.expect<LT::Comma, LT::RightRound>() == LT::RightRound {}) {
				//no more rule setting
				break;
			}
		}

		if (lexer.expect<LT::Comma, LT::RightCurly>() == LT::RightCurly {}) {
			//no more rule
			break;
		}
	}
}

static void processGroup(STPTDLLexer& lexer, STPResult& result) {
	//the declared type of this group
	const string_view group_type = **lexer.expect<TDLIdentifier>();
	lexer.expect<LT::LeftCurly>();

	//record all textures to be added to a new group,
	//always make sure data being parsed are valid before adding to the parsing memory.
	vector<string_view> texture_in_group;
	//for each group
	while (true) {
		//clear old data
		texture_in_group.clear();

		//for each texture name assigned to this group
		while (true) {
			const string_view& assignedTexture = texture_in_group.emplace_back(**lexer.expect<TDLIdentifier>());
			checkTextureDeclared(lexer, result, assignedTexture);

			if (lexer.expect<LT::Comma, LT::Colon>() == LT::Colon {}) {
				//no more texture to be assigned to this group
				break;
			}
		}
		lexer.expect<LT::Equal>();

		//beginning of a group definition tuple
		lexer.expect<LT::LeftRound>();
		if (group_type == "view") {
			STPTextureDatabase::STPViewGroupDescription& view = result.DeclaredViewGroup.emplace_back();

			view.PrimaryScale = lexer.expect<TDLNumber>()->to<unsigned int>();
			lexer.expect<LT::Comma>();
			view.SecondaryScale = lexer.expect<TDLNumber>()->to<unsigned int>();
			lexer.expect<LT::Comma>();
			view.TertiaryScale = lexer.expect<TDLNumber>()->to<unsigned int>();
		} else {
			ostringstream msg;
			msg << "Group type \'" << group_type << "\' is not recognised." << endl;
			lexer.throwSyntaxError(msg.str(), "unrecognised group type");
		}
		//end of a group definition tuple
		lexer.expect<LT::RightRound>();

		//assign texture with group index
		std::for_each(texture_in_group.cbegin(), texture_in_group.cend(),
			[&texture_table = result.DeclaredTexture, view_group_index = result.DeclaredViewGroup.size() - 1u](
				const auto& name) {
				//assign all texture to be added with the group just parsed
				texture_table[name] = view_group_index;
			});

		if (lexer.expect<LT::Comma, LT::RightCurly>() == LT::RightCurly {}) {
			//end of group
			break;
		}
	}
}

STPResult STPTextureDefinitionLanguage::read(const char* const source, const string_view& source_name) {
	STPTDLLexer lexer(source, TDLParserName, source_name);

	STPResult result;
	//start doing lexical analysis and parsing
	while (true) {
		//check while identifier is it
		if (lexer.expect<LT::Hash, LT::Null>() == LT::Null {}) {
			//end of file
			break;
		}
		const string_view operation = **lexer.expect<TDLIdentifier>();

		//depends on operations, we process them differently
		if (operation == "texture") {
			processTexture(lexer, result);
		} else if (operation == "rule") {
			processRule(lexer, result);
		} else if (operation == "group") {
			processGroup(lexer, result);
		} else {
			//invalid operation
			ostringstream msg;
			msg << "Directive \'" << operation << "\' is undefined by Texture Definition Language." << endl;
			lexer.throwSyntaxError(msg.str(), "unknown operation");
		}
	}

	//semantics check
	for_each(result.DeclaredTexture.cbegin(), result.DeclaredTexture.cend(), [](const auto& texture) {
		const auto& [texture_name, view_group_index] = texture;
		if (view_group_index == STPResult::UnreferencedIndex) {
			//this index has no corresponding view group
			ostringstream msg;
			msg << "View group reference for \'" << texture_name << "\' is undefined." << endl;
			throw STP_PARSER_SEMANTIC_ERROR_CREATE(msg.str(), TDLParserName, "undefined view group reference");
		}
	});

	return result;
}

STPResult::STPTextureVariable STPTextureDefinitionLanguage::STPResult::load(STPDiversity::STPTextureDatabase& database) const {
	//prepare variable dictionary for return
	STPTextureVariable varDic;
	varDic.reserve(this->DeclaredTexture.size());

	namespace TI = SuperTerrainPlus::STPDiversity::STPTextureInformation;
	//to convert view group index to corresponded ID in the database
	unique_ptr<TI::STPViewGroupID[]> ViewGroupIDLookup = make_unique<TI::STPViewGroupID[]>(this->DeclaredViewGroup.size());
	//add texture view group
	std::transform(this->DeclaredViewGroup.cbegin(), this->DeclaredViewGroup.cend(), ViewGroupIDLookup.get(), [&database](const auto& view_desc) {
		return database.addViewGroup(view_desc);
	});

	//requesting texture
	//assign each variable with those texture ID
	for_each(this->DeclaredTexture.cbegin(), this->DeclaredTexture.cend(), [&database, &varDic, id_lookup = ViewGroupIDLookup.get()](const auto& texture) {
		const auto& [texture_name, view_group_index] = texture;
		//we have already checked the view group index at the end of parsing, so it must be valid
		const TI::STPViewGroupID view_group_id = id_lookup[view_group_index];
		varDic.try_emplace(texture_name, database.addTexture(view_group_id, make_optional(texture_name)), view_group_id);	
	});

	//add splat rules into the database
	//when we were parsing the TDL, we have already checked all used texture are declared, so we are sure textureName
	//can be located in the dictionary
	STPTextureDatabase::STPTextureSplatBuilder& splat_builder = database.splatBuilder();
	for_each(this->Altitude.cbegin(), this->Altitude.cend(), [&splat_builder, &var = as_const(varDic)](const auto& altitude) {
		const auto& [sample, ub, textureName] = altitude;
		//one way to call function using tuple is std::apply, however we need to replace textureName with textureID in this database.
		splat_builder.addAltitude(sample, ub, var.find(textureName)->second.first);
	});
	for_each(this->Gradient.cbegin(), this->Gradient.cend(), [&splat_builder, &var = as_const(varDic)](const auto& gradient) {
		const auto& [sample, minG, maxG, lb, ub, textureName] = gradient;
		splat_builder.addGradient(sample, minG, maxG, lb, ub, var.find(textureName)->second.first);
	});

	return varDic;
}