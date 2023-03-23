#include <SuperAlgorithm+Host/Parser/STPTextureDefinitionLanguage.h>
#include <SuperAlgorithm+Host/Parser/Framework/STPLexer.h>

//Error
#include <SuperTerrain+/Exception/STPParserError.h>

#include <sstream>

#include <algorithm>
#include <memory>
#include <utility>

using std::string_view;
using std::ostringstream;
using std::vector;

using std::unique_ptr;
using std::make_unique;
using std::as_const;

using std::for_each;

using std::endl;

using SuperTerrainPlus::STPDiversity::Sample;
using SuperTerrainPlus::STPDiversity::STPTextureDatabase;

using namespace SuperTerrainPlus::STPAlgorithm;

//the name of our TDL parser
constexpr static char TDLParserName[] = "SuperTerrain+ Texture Definition Language";

//define our TDL lexer
namespace {
	namespace RL = STPRegularLanguage;
	namespace CC = RL::STPCharacterClass;

	template<const string_view& S>
	using Lit = RL::Literal<S>;

	constexpr string_view DirControl = "#", DirTexture = "texture", DirGroup = "group", DirRule = "rule",
		GroupView = "view",
		RuleAlt = "altitude", RuleGra = "gradient",
		TexStart = "[", TexEnd = "]", DirStart = "{", DirEnd = "}", ParamStart = "(", ParamEnd = ")",
		OpAssignment = ":=", OpRightArrow = "->", OpComma = ",",
		NumDecimal = ".";

	//ignore
	using SkipSpace = CC::Class<CC::Atomic<' '>, CC::Atomic<'\n'>, CC::Atomic<'\t'>>;
	
	STP_LEXER_CREATE_TOKEN_EXPRESSION(SpaceNewlineInsensitive, 0xABCDu, Discard, SkipSpace);

	//directive
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveControl, 0x00u, Consume, Lit<DirControl>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveTexture, 0x01u, Consume, Lit<DirTexture>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveGroup, 0x02u, Consume, Lit<DirGroup>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveRule, 0x03u, Consume, Lit<DirRule>);
	//sub directive
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveGroupView, 0x0Au, Consume, Lit<GroupView>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveRuleAltitude, 0x0Bu, Consume, Lit<RuleAlt>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveRuleGradient, 0x0Cu, Consume, Lit<RuleGra>);

	//block scope
	STP_LEXER_CREATE_TOKEN_EXPRESSION(TextureBlockStart, 0x10u, Consume, Lit<TexStart>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(TextureBlockEnd, 0x11u, Consume, Lit<TexEnd>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveBlockStart, 0x12u, Consume, Lit<DirStart>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DirectiveBlockEnd, 0x13u, Consume, Lit<DirEnd>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(ParameterBlockStart, 0x14u, Consume, Lit<ParamStart>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(ParameterBlockEnd, 0x15u, Consume, Lit<ParamEnd>);

	//operator
	STP_LEXER_CREATE_TOKEN_EXPRESSION(Assignment, 0x20u, Consume, Lit<OpAssignment>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(MapTo, 0x21u, Consume, Lit<OpRightArrow>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(Separator, 0x22u, Consume, Lit<OpComma>);

	using RL::STPQuantifier::Maybe;
	using RL::STPQuantifier::StrictMany;
	//special matcher
	template<char Suf>
	using MaybeSuffix = Maybe<CC::Class<CC::Atomic<Suf>>>;//numeric suffix is optional
	using MatchIdentifier =
		StrictMany<
			CC::Class<
				CC::Atomic<'_'>,
				CC::Range<'a', 'z'>
			>
		>;
	using Number =
		StrictMany<
			CC::Class<
				CC::Range<'0', '9'>
			>
		>;
	using MatchInteger = RL::Sequence<Number, MaybeSuffix<'u'>>;
	//we force to use standard floating point declaration to make things easier, such that format such as .1 or 0f are not allowed
	using MatchFloat = RL::Sequence<Number, Lit<NumDecimal>, Number, MaybeSuffix<'f'>>;
	
	STP_LEXER_CREATE_TOKEN_EXPRESSION(TDLIdentifier, 0x30u, Consume, MatchIdentifier);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(TDLUnsignedInteger, 0x31u, Consume, MatchInteger);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(TDLFloat, 0x32u, Consume, MatchFloat);

	//state
	STP_LEXER_CREATE_LEXICAL_STATE(TDLMainState, 0x88u,
		SpaceNewlineInsensitive,
		//---
		DirectiveControl, DirectiveTexture, DirectiveGroup, DirectiveRule,
		DirectiveGroupView, DirectiveRuleAltitude, DirectiveRuleGradient,
		//---
		TextureBlockStart, TextureBlockEnd, DirectiveBlockStart, DirectiveBlockEnd, ParameterBlockStart, ParameterBlockEnd,
		//---
		Assignment, MapTo, Separator,
		//---
		TDLIdentifier, TDLUnsignedInteger, TDLFloat);

	//the lexer for TDL
	typedef STPLexer<TDLMainState> STPTDLLexer;
}

using STPTextureDefinitionLanguage::STPResult;

static void checkTextureDeclared(STPResult& result, const string_view& texture) {
	if (result.DeclaredTexture.find(texture) == result.DeclaredTexture.cend()) {
		//texture variable not found, throw error
		ostringstream msg;
		msg << "Texture \'" << texture << "\' is not declared before it is being referenced." << endl;
		throw STP_PARSER_SEMANTIC_ERROR_CREATE(msg.str(), TDLParserName, "unknown texture");
	}
}

static void processTexture(STPTDLLexer& lexer, STPResult& result) {
	//declare some texture variables for texture ID
	lexer.expect<TextureBlockStart>();

	while (true) {
		const string_view textureName = **lexer.expect<TDLIdentifier>();
		//found a texture, store it
		//initially the texture has no view information
		result.DeclaredTexture.emplace(textureName, STPResult::UnreferencedIndex);

		if (lexer.expect<Separator, TextureBlockEnd>() == TextureBlockEnd {}) {
			//no more texture
			break;
		}
		//a comma means more texture are coming...
	}
}

static void processRule(STPTDLLexer& lexer, STPResult& result) {
	//define a rule
	const STPTDLLexer::STPToken rule_type = lexer.expect<DirectiveRuleAltitude, DirectiveRuleGradient>();
	lexer.expect<DirectiveBlockStart>();

	while (true) {
		//we got a sample ID
		const Sample rule4Sample = lexer.expect<TDLUnsignedInteger>()->to<Sample>();
		lexer.expect<Assignment>();
		lexer.expect<ParameterBlockStart>();

		//start parsing rule
		while (true) {
			//check which type of type we are parsing
			if (rule_type == DirectiveRuleAltitude {}) {
				const float altitude = lexer.expect<TDLFloat>()->to<float>();
				lexer.expect<MapTo>();
				const string_view map2Texture = **lexer.expect<TDLIdentifier>();
				checkTextureDeclared(result, map2Texture);

				//store an altitude rule
				result.Altitude.emplace_back(rule4Sample, altitude, map2Texture);
			} else if (rule_type == DirectiveRuleGradient {}) {
				const float minG = lexer.expect<TDLFloat>()->to<float>();
				lexer.expect<Separator>();
				const float maxG = lexer.expect<TDLFloat>()->to<float>();
				lexer.expect<Separator>();
				const float LB = lexer.expect<TDLFloat>()->to<float>();
				lexer.expect<Separator>();
				const float UB = lexer.expect<TDLFloat>()->to<float>();
				lexer.expect<MapTo>();
				const string_view map2Texture = **lexer.expect<TDLIdentifier>();
				checkTextureDeclared(result, map2Texture);

				//store a gradient rule
				result.Gradient.emplace_back(rule4Sample, minG, maxG, LB, UB, map2Texture);
			}
			//else? the lexer will handle the exception for us :)

			if (lexer.expect<Separator, ParameterBlockEnd>() == ParameterBlockEnd {}) {
				//no more rule setting
				break;
			}
		}

		if (lexer.expect<Separator, DirectiveBlockEnd>() == DirectiveBlockEnd {}) {
			//no more rule
			break;
		}
	}
}

static void processGroup(STPTDLLexer& lexer, STPResult& result) {
	//the declared type of this group
	const STPTDLLexer::STPToken group_type = lexer.expect<DirectiveGroupView>();
	lexer.expect<DirectiveBlockStart>();

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
			checkTextureDeclared(result, assignedTexture);

			if (lexer.expect<Separator, Assignment>() == Assignment {}) {
				//no more texture to be assigned to this group
				break;
			}
		}

		//beginning of a group definition tuple
		lexer.expect<ParameterBlockStart>();
		if (group_type == DirectiveGroupView {}) {
			STPTextureDatabase::STPViewGroupDescription& view = result.DeclaredViewGroup.emplace_back();

			view.PrimaryScale = lexer.expect<TDLUnsignedInteger>()->to<unsigned int>();
			lexer.expect<Separator>();
			view.SecondaryScale = lexer.expect<TDLUnsignedInteger>()->to<unsigned int>();
			lexer.expect<Separator>();
			view.TertiaryScale = lexer.expect<TDLUnsignedInteger>()->to<unsigned int>();
		}
		//end of a group definition tuple
		lexer.expect<ParameterBlockEnd>();

		//assign texture with group index
		std::for_each(texture_in_group.cbegin(), texture_in_group.cend(),
			[&texture_table = result.DeclaredTexture, view_group_index = result.DeclaredViewGroup.size() - 1u](
				const auto& name) {
				//assign all texture to be added with the group just parsed
				texture_table[name] = view_group_index;
			});

		if (lexer.expect<Separator, DirectiveBlockEnd>() == DirectiveBlockEnd {}) {
			//end of group
			break;
		}
	}
}

STPResult STPTextureDefinitionLanguage::read(const string_view& source, const string_view& source_name) {
	STPTDLLexer lexer(source, TDLParserName, source_name);

	STPResult result;
	//start doing lexical analysis and parsing
	while (true) {
		//check while identifier is it
		if (lexer.expect<DirectiveControl, STPLexical::EndOfSequence>() == STPLexical::EndOfSequence {}) {
			//end of file
			break;
		}
		const STPTDLLexer::STPToken operation = lexer.expect<DirectiveTexture, DirectiveRule, DirectiveGroup>();

		//depends on operations, we process them differently
		if (operation == DirectiveTexture {}) {
			processTexture(lexer, result);
		} else if (operation == DirectiveRule {}) {
			processRule(lexer, result);
		} else if (operation == DirectiveGroup {}) {
			processGroup(lexer, result);
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
	for_each(this->DeclaredTexture.cbegin(), this->DeclaredTexture.cend(),
		[&database, &varDic, id_lookup = ViewGroupIDLookup.get()](const auto& texture) {
		const auto& [texture_name, view_group_index] = texture;
		//we have already checked the view group index at the end of parsing, so it must be valid
		const TI::STPViewGroupID view_group_id = id_lookup[view_group_index];
		varDic.try_emplace(texture_name, database.addTexture(view_group_id, texture_name), view_group_id);	
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