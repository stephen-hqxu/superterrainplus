#include <SuperAlgorithm+Host/Parser/STPINIParser.h>
#include <SuperAlgorithm+Host/Parser/Framework/STPLexer.h>

#include <cassert>

using std::string_view;

using namespace SuperTerrainPlus::STPAlgorithm;

constexpr static char INIReaderName[] = "SuperTerrain+ MS INI Reader";

//define an INI lexer
namespace {
	namespace RL = STPRegularLanguage;
	namespace CC = RL::STPCharacterClass;

	constexpr string_view SecStart = "[", SecEnd = "]", KVSeparator = "=", Newline = "\n";

	using EmptyLine =
		CC::Class<
			CC::Atomic<' '>,
			CC::Atomic<'\n'>
		>;
	using CmtStart = CC::Class<CC::Atomic<'#'>, CC::Atomic<';'>>;
	using KVStart =
		//the first character should not be any of the control symbol
		CC::Class<
			CC::Except<
				CC::Atomic<'['>,
				CC::Atomic<'#'>,
				CC::Atomic<';'>,
				CC::Atomic<'='>
			>
		>;
	using MatchSecName =
		RL::STPQuantifier::StrictMany<
			CC::Class<
				CC::Except<
					CC::Atomic<'\n'>,
					CC::Atomic<']'>
				>
			>
		>;
	using MatchKey =
		RL::STPQuantifier::MaybeMany<
			CC::Class<
				CC::Except<
					CC::Atomic<'='>,
					CC::Atomic<'\n'>
				>
			>
		>;
	using NextLine = RL::Literal<Newline>;
	using NotNextLine = CC::Class<CC::Except<CC::Atomic<'\n'>>>;

	/* --------------------------------------------- main state ------------------------------------------------- */
	//this state detects what type of line we are currently at
	STP_LEXER_CREATE_TOKEN_EXPRESSION(MasterSkipEmptyLine, 0xFEDCu, Discard, EmptyLine);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(SectionLineStart, 0x00u, Consume, RL::Literal<SecStart>, 0xAAu);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(KeyValueLineStart, 0x10u, Collect, KVStart, 0xBBu);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(CommentLineStart, 0x20u, Discard, CmtStart, 0xCCu);

	/* ---------------------------------------- section name matching ------------------------------------------------ */
	STP_LEXER_CREATE_TOKEN_EXPRESSION(SectionName, 0x01u, Consume, MatchSecName);
	STP_LEXER_CREATE_TOKEN_EXPRESSION(SectionLineClosing, 0x02u, Consume, RL::Literal<SecEnd>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(SectionLineEnd, 0x03u, Consume, NextLine, 0x00u);//end of section declaration line

	/* ----------------------------------------- key-value matching ----------------------------------------------- */
	STP_LEXER_CREATE_TOKEN_EXPRESSION(KeyName, 0x11u, Consume, MatchKey);
	//the key lexeme will contain the delimiter at the end, remember to remove it during parsing
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(KeyValueSeparator, 0x12u, Consume, RL::Literal<KVSeparator>, 0xDDu);//switch to value state
	STP_LEXER_CREATE_TOKEN_EXPRESSION(ValueName, 0x13u, Consume, RL::STPQuantifier::MaybeMany<NotNextLine>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(KeyValueLineEnd, 0x14u, Consume, NextLine, 0x00u);//end of key-value line

	/* ----------------------------------------- comment matching ---------------------------------------------------- */
	STP_LEXER_CREATE_TOKEN_EXPRESSION(DiscardComment, 0x21u, Collect, NotNextLine);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(CommentLineEnd, 0x22u, Discard, NextLine, 0x00u);//end of comment line

	/* ----------------------------------------------------------------------------------------------------------------- */
	STP_LEXER_CREATE_LEXICAL_STATE(INIMasterState, 0x00u, MasterSkipEmptyLine, SectionLineStart, KeyValueLineStart, CommentLineStart);
	STP_LEXER_CREATE_LEXICAL_STATE(INISectionState, 0xAAu, SectionName, SectionLineClosing, SectionLineEnd);
	STP_LEXER_CREATE_LEXICAL_STATE(INIKeyState, 0xBBu, KeyName, KeyValueSeparator);
	//--- sub-state to match value
	STP_LEXER_CREATE_LEXICAL_STATE(INIValueState, 0xDDu, ValueName, KeyValueLineEnd);
	//---
	STP_LEXER_CREATE_LEXICAL_STATE(INICommentState, 0xCCu, DiscardComment, CommentLineEnd);

	typedef STPLexer<INIMasterState, INISectionState, INIKeyState, INIValueState, INICommentState> STPINILexer;
}

//Remove white space from both ends
static string_view doubleTrim(const string_view& s) noexcept {
	constexpr static string_view WhiteSpace = " \f\n\r\t\v";

	//Remove all leading white space
	constexpr static auto ltrim = [](const string_view& s) constexpr noexcept -> string_view {
		const size_t start = s.find_first_not_of(WhiteSpace);
		return start == string_view::npos ? string_view() : s.substr(start);
	};
	//Remove all trailing white space
	constexpr static auto rtrim = [](const string_view& s) constexpr noexcept -> string_view {
		const size_t end = s.find_last_not_of(WhiteSpace);
		return end == string_view::npos ? string_view() : s.substr(0u, end + 1u);
	};

	return rtrim(ltrim(s));
}

using STPINIParser::STPINIReaderResult;

STPINIReaderResult STPINIParser::read(const string_view& ini, const string_view& ini_name) {
	STPINILexer lexer(ini, ini_name, INIReaderName);

	STPINIReaderResult result;
	size_t currentSecIdx = std::numeric_limits<size_t>::max(), currentPropIdx = 0u;
	const auto addSection = [&result, &secIdx = currentSecIdx, &propIdx = currentPropIdx](const string_view& sec_name) -> STPINIData::STPINISectionView& {
		auto& [storage, section_order, property_order] = result;

		//add new order table
		//section index starts from 0, and all properties added later will use this section index
		section_order.try_emplace(sec_name, ++secIdx);
		property_order.emplace_back();
		//reset property counter in a new section
		propIdx = 0u;

		//insert a new section
		return storage.try_emplace(sec_name).first->second;
	};

	//start with global section, which is unnamed
	STPINIData::STPINISectionView* current_sec = &addSection("");
	while (true) {
		//check what type of line we are currently at
		const STPINILexer::STPToken line_tok = lexer.expect<SectionLineStart, KeyName, STPLexical::EndOfSequence>();

		if (line_tok == SectionLineStart {}) {
			//a new section, need to find the section name
			const string_view section_name = doubleTrim(**lexer.expect<SectionName>());
			if (section_name.empty()) {
				lexer.throwSyntaxError("The section name is empty", "empty section");
			}

			//start a new section
			current_sec = &addSection(section_name);

			//closing section
			lexer.expect<SectionLineClosing>();
		} else if (line_tok == KeyName {}) {
			//it is a key-value property string
			//key
			const string_view key_name = doubleTrim(**line_tok);
			//by our lexer grammar this should not happen
			assert(!key_name.empty());

			lexer.expect<KeyValueSeparator>();

			//value
			const string_view value_name = doubleTrim(**lexer.expect<ValueName>());
			//similarly
			assert(!value_name.empty());

			current_sec->try_emplace(key_name, value_name);
			result.PropertyOrder[currentSecIdx].try_emplace(key_name, currentPropIdx++);
		} else {
			//end of parsing input
			assert(line_tok == STPLexical::EndOfSequence {});
			break;
		}
		
		//next line
		//if we are at the end of input, the lexer will always return the EOS special token
		lexer.expect<SectionLineEnd, KeyValueLineEnd, STPLexical::EndOfSequence>();
	}
	return result;
}