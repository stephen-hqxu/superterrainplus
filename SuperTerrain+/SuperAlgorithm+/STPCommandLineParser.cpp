#include <SuperAlgorithm+/Parser/STPCommandLineParser.h>
#include <SuperAlgorithm+/Parser/Framework/STPLexer.h>

#include <SuperTerrain+/Exception/STPValidationFailed.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//Hash
#include <SuperTerrain+/Utility/STPHashCombine.h>

//Container
#include <stack>
#include <queue>
#include <unordered_set>
#include <unordered_map>

//Stream
#include <iomanip>
#include <ios>
#include <sstream>

#include <cassert>
#include <functional>

using std::vector;
using std::stack;
using std::queue;
using std::unordered_set;
using std::unordered_map;
using std::pair;
using std::optional;

using std::string;
using std::string_view;
using std::ostringstream;
using std::ostream;
using std::streamsize;

using std::count_if;
using std::endl;
using std::for_each;
using std::forward;
using std::make_optional;
using std::make_pair;
using std::setw;

using namespace SuperTerrainPlus::STPAlgorithm;

//A special delimiter to separate each argument,
//this unit separator is defined by ASCII as a legit data separator.
constexpr static char ArgumentDelimiter = '\x1F';

string STPCommandLineParser::encode(const int argc, const char* const* const argv) {
	ostringstream processed;
	for (int i = 0; i < argc; i++) {
		//insert a delimiter at the end of each argument
		processed << argv[i] << ArgumentDelimiter;
	}
	//notice we will have a delimiter at the end of the string, before the null
	return processed.str();
}

constexpr static char CmdParserName[] = "SuperTerrain+ Command Line Parser";
#define CMD_PARSER_SEMANTIC_ERROR(MSG, TITLE) STP_PARSER_SEMANTIC_ERROR_CREATE(MSG, CmdParserName, TITLE)

//define command line lexer
namespace {
	namespace RL = STPRegularLanguage;
	namespace CC = RL::STPCharacterClass;

	constexpr string_view ShortOptionKey = "-", LongOptionKey = "--", OptionEndSymbol = "=";

	using OptionCharacterLower = CC::Range<'a', 'z'>;
	using OptionCharacterUpper = CC::Range<'A', 'Z'>;
	using DelimiterCharacter = CC::Atomic<ArgumentDelimiter>;
	//---
	using ShortOptionString = CC::Class<OptionCharacterLower, OptionCharacterUpper>;
	using LongOptionString =
		RL::STPQuantifier::StrictMany<
			CC::Class<
				CC::Atomic<'-'>,
				OptionCharacterLower,
				OptionCharacterUpper
			>
		>;

	/* --------------------------------------------- global ----------------------------------------------- */
	STP_LEXER_CREATE_TOKEN_EXPRESSION(CommandSeparator, 0xA0u, Consume, CC::Class<DelimiterCharacter>);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(ShortOptionControl, 0xA1u, Consume, RL::Literal<ShortOptionKey>, 0x7Eu);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(LongOptionControl, 0xA2u, Consume, RL::Literal<LongOptionKey>, 0x7Fu);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(ValueControl, 0xA3u, Collect, RL::Any, 0x6Fu);

	/* ---------------------------------------------- value ----------------------------------------------------- */
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(ValueName, 0xAFu, Consume,
		RL::STPQuantifier::MaybeMany<CC::Class<CC::Except<DelimiterCharacter>>>, 0x66u);

	/* ---------------------------------------------- option -------------------------------------------------- */
	//--- long option
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(LongOptionName, 0xBFu, Consume, LongOptionString, 0x7Eu);
	//--- regular option, this allows putting multiple short options together without specifying control symbols
	STP_LEXER_CREATE_TOKEN_EXPRESSION(ShortOptionName, 0xB1u, Consume, ShortOptionString);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(OptionEndCompact, 0xB2u, Consume, RL::Literal<OptionEndSymbol>, 0x66u);
	STP_LEXER_CREATE_TOKEN_EXPRESSION_SWITCH_STATE(OptionEndSpread, 0xB3u, Consume, CC::Class<DelimiterCharacter>, 0x66u);

	/* -------------------------------------------------------------------------------------------------------------------- */
	STP_LEXER_CREATE_LEXICAL_STATE(CmdGlobalState, 0x66u, CommandSeparator, ShortOptionControl, LongOptionControl, ValueControl);
	STP_LEXER_CREATE_LEXICAL_STATE(CmdValueState, 0x6Fu, ValueName);
	STP_LEXER_CREATE_LEXICAL_STATE(CmdLongOptionState, 0x7Fu, LongOptionName);
	STP_LEXER_CREATE_LEXICAL_STATE(CmdRegularOptionState, 0x7Eu, ShortOptionName, OptionEndCompact, OptionEndSpread);

	typedef STPLexer<CmdGlobalState, CmdValueState, CmdLongOptionState, CmdRegularOptionState> STPCmdLexer;

	//The type for the table storing option names.
	//The boolean specifies if it is a short option, the string specifies its name.
	typedef pair<bool, string_view> STPOptionTypeName;
	/**
	 * @brief A hash function for option name and the type of the option.
	*/
	struct STPHashOptionTypeName {
	public:

		inline size_t operator()(const STPOptionTypeName& type_name) const noexcept {
			const auto& [is_short, opt_name] = type_name;

			size_t value = 0u;
			SuperTerrainPlus::STPHashCombine::combine(value, is_short, opt_name);
			return value;
		}

	};
}

namespace STPInternal = STPCommandLineParser::STPInternal;

void STPInternal::STPBaseOption::throwConversionError(const char* const custom, const STPReceivedArgument& rx_arg) {
	ostringstream error;

	error << custom << endl;
	error << "The following arguments received from the command line cannot be converted:" << endl;
	for (const auto& arg : rx_arg) {
		error << '\t' << *arg << endl;
	}
	error << "Please make sure the binding variable(s) is/are valid for the expected argument received" << endl;

	throw CMD_PARSER_SEMANTIC_ERROR(error.str(), "argument conversion error");
}

//Positional arguments recognised from the command line.
typedef queue<STPStringViewAdaptor> STPReceivedPositional;
//Given the option name, either short or long, return the pointer to this option.
//This includes options in the group. Subcommand options are ignored.
typedef unordered_map<STPOptionTypeName, const STPInternal::STPBaseOption*, STPHashOptionTypeName> STPOptionTable;
//A subset of the option table, storing positional options based on their precedence.
typedef vector<const STPInternal::STPBaseOption*> STPPositionalOptionTable;
//The pointer of the option configuration of the next option,
//and a boolean value specifies if this option uses DSV argument format.
typedef optional<pair<const STPInternal::STPBaseOption*, bool>> STPOptionResult;

/**
 * @brief Apply a function for each of the non-empty option name.
 * @tparam Func A function that takes a bool and a pointer to constant string view as the non-empty name.
 * The bool value indicates if this option is a short name.
 * @param option The option whose names should be checked and applied.
 * @param f The function to be applied.
*/
template<class Func>
inline static void applyOptionName(const STPInternal::STPBaseOption& option, Func&& f) {
	if (!option.ShortName.empty()) {
		forward<Func>(f)(true, option.ShortName);
	}
	if (!option.LongName.empty()) {
		forward<Func>(f)(false, option.LongName);
	}
}

/**
 * @brief Print an option by its name to the stream.
 * @param stream The stream to be printed to.
 * @param option The option to be printed.
 * @param separator The separator to separate the long name and short name, if both are defined.
 * @return The stream.
*/
static ostream& displayOption(ostream& stream, const STPInternal::STPBaseOption& option, const char separator = ',') {
	//use separator if both option names are defined
	const bool useSeparator = !option.ShortName.empty() && !option.LongName.empty();

	if (!option.ShortName.empty()) {
		stream << '-' << option.ShortName;
	}
	if (useSeparator) {
		stream << separator;
	}
	if (!option.LongName.empty()) {
		stream << "--" << option.LongName;
	}

	return stream;
}

void STPCommandLineParser::validate(const STPInternal::STPBaseCommand& command) {
	//precondition
	STP_ASSERTION_VALIDATION(!command.isGroup(), "The top level command must not be a group");

	const auto validateCurrentOption = [](const STPInternal::STPBaseOption& option) -> void {
		const auto messageOptionError = [&option](const char* const extra_msg) -> string {
			ostringstream msg;
			msg << "Validation failure for option \'";
			displayOption(msg, option) << '\'' << endl;
			msg << extra_msg;
			
			return msg.str();
		};

		const auto isNameValid = [](const size_t matchLength, const string_view& name) -> bool {
			//we need to make sure the entire option name is an exact match of the expression
			return name.empty() || (matchLength != RL::NoMatch && matchLength == name.length());
		};
		const string_view &shortName = option.ShortName,
			&longName = option.LongName;
		const size_t shortMatch = ShortOptionString::match(shortName),
			longMatch = LongOptionString::match(longName);

		STP_ASSERTION_VALIDATION(!(shortName.empty() && longName.empty()),
			"At least one of the option name should be specified");
		STP_ASSERTION_VALIDATION(isNameValid(shortMatch, shortName) && isNameValid(longMatch, longName),
			messageOptionError("The format of one of the option name is invalid"));

		const auto [min, max] = option.ArgumentCount;
		STP_ASSERTION_NUMERIC_DOMAIN(min <= max, messageOptionError("The minimum number of argument must be no greater than the maximum"));
	};
	//does not include descendant commands and options
	const auto validateCurrentCommand = [root = &command](const STPInternal::STPBaseCommand& command) -> void {
		const auto messageCommandError = [&command](const char* const extra_msg) -> string {
			ostringstream msg;
			msg << "Validation failure for command: \'" << command.Name << '\'' << endl;
			msg << extra_msg;

			return msg.str();
		};

		STP_ASSERTION_VALIDATION(!command.Name.empty(), "Name of any command or group must be non-empty");
		if (command.isSubcommand() && &command != root) {
			//root command does not matter, because we don't need to specify its name from the command line
			//use the same name convention for the subcommand as the long option name
			const size_t subcommandNameMatch = LongOptionString::match(command.Name);
			STP_ASSERTION_VALIDATION(subcommandNameMatch != RL::NoMatch && subcommandNameMatch == command.Name.length(),
				messageCommandError("The format of the subcommand name is invalid"));
		}

		const auto [min, max] = command.OptionCount;
		STP_ASSERTION_NUMERIC_DOMAIN(min <= max, messageCommandError("The minimum number of option must be no greater than the maximum"));
	};
	//detect if any option name is duplicate within the current subcommand level
	//this also includes all groups in the current level
	unordered_set<string_view> duplicateOption;
	//detect if any subcommand name is duplicate within the same level
	unordered_set<string_view> duplicateSubcommand;
	//BFS is more cache friendly than DFS
	queue<const STPInternal::STPBaseCommand*> subcommandQueue, groupQueue;

	//error message creator
	const auto messageSubcommandInGroup = [](const string_view& root_name, const string_view& group_name,
		const string_view& problematic_subcommand_name) -> string {
		ostringstream err;
		err << "A group is not allowed to contain any subcommand; but in command \'" << root_name << "\', a group \'"
			<< group_name << "\' has a subcommand named \'" << problematic_subcommand_name << '\'' << endl;
		return err.str();
	};
	const auto messageSubcommandDuplication = [](const string_view& parent_sub_name, const string_view& duplicate_sub_name) -> string {
		ostringstream err;
		err << "Subcommand \'" << duplicate_sub_name << "\' defined under command \'" << parent_sub_name
			<< "\' is redefined" << endl;
		return err.str();
	};

	//loop through all commands
	subcommandQueue.push(&command);
	while (!subcommandQueue.empty()) {
		const STPInternal::STPBaseCommand& currentSubcommand = *subcommandQueue.front();
		subcommandQueue.pop();

		//use this subcommand as a new root, clear option name for the current subcommand hierarchy
		duplicateOption.clear();
		
		//the current subcommand, recursively check for its groups and options
		assert(groupQueue.empty());
		groupQueue.push(&currentSubcommand);
		while (!groupQueue.empty()) {
			const STPInternal::STPBaseCommand& currentGroup = *groupQueue.front();
			groupQueue.pop();

			//options in a subcommand
			for (const auto* const opt : currentGroup.option()) {
				validateCurrentOption(*opt);

				//add option to table, or throw exception if name is duplicate
				const auto addToTable = [&duplicateOption, &currentGroup](bool, const string_view& name) -> void {
					const auto messageDuplicateOptionName = [](const string_view& option_name, const string_view& subcommand_name) -> string {
						ostringstream err;
						err << "Option \'" << option_name << "\' is not unique in command \'" << subcommand_name << '\'' << endl;
						return err.str();
					};

					const bool isOptionUnique = duplicateOption.emplace(name).second;
					STP_ASSERTION_VALIDATION(isOptionUnique, messageDuplicateOptionName(name, currentGroup.Name));
				};
				applyOptionName(*opt, addToTable);
			}

			//groups in a subcommand
			//validate current node
			validateCurrentCommand(currentGroup);
			for (const auto* const cmd : currentGroup.command()) {
				//this branch is only possible if the parent is root (so a subcommand), otherwise it is an error
				//the child can be anything if parent is a subcommand
				//the child can only be a group is parent is a group also
				STP_ASSERTION_VALIDATION(currentGroup.isSubcommand() || cmd->isGroup(),
					messageSubcommandInGroup(currentSubcommand.Name, currentGroup.Name, cmd->Name));

				if (cmd->isGroup()) {
					//ignore subcommand, only consider group
					groupQueue.push(cmd);
				}
			}
		}

		//clear subcommand name for the current level
		duplicateSubcommand.clear();
		for (const auto* const sub : currentSubcommand.command()) {
			if (!sub->isSubcommand()) {
				continue;
			}
			//check for name duplication
			const bool isSubcommandUnique = duplicateSubcommand.emplace(sub->Name).second;
			STP_ASSERTION_VALIDATION(isSubcommandUnique, messageSubcommandDuplication(currentSubcommand.Name, sub->Name));

			//record this subcommand so next time we can check its children
			subcommandQueue.push(sub);
		}
	}
}

/**
 * @brief Expect tokens at the start of a command section.
 * @param lexer The lexer.
 * @return The expected token.
*/
inline static STPCmdLexer::STPToken expectCommandSectionStart(STPCmdLexer& lexer) {
	return lexer.expect<ShortOptionControl, LongOptionControl, ValueName, STPLexical::EndOfSequence>();
}

/**
 * @brief Locate the subcommand where the parsing should start.
 * @param lexer The lexer.
 * @param root The root of the command line command configuration.
 * @return The path of the command tree corresponds to the subcommands provided in the command line,
 * where the last subcommand is where parsing should start; and the last token returned by the lexer.
*/
static pair<vector<const STPInternal::STPBaseCommand*>, STPCmdLexer::STPToken> findCommand(
	STPCmdLexer& lexer, const STPInternal::STPBaseCommand& root) {
	vector<const STPInternal::STPBaseCommand*> commandPath;
	commandPath.push_back(&root);

	const auto createResult = [&commandPath](const STPCmdLexer::STPToken& token) noexcept -> auto {
		return make_pair(std::move(commandPath), token);
	};
	//Basically we want to consume as many subcommand token as many as we can,
	//this is done by travelling down the hierarchy until we hit something that is not a subcommand.
	while (true) {
		const STPCmdLexer::STPToken& command_tok = expectCommandSectionStart(lexer);
		if (command_tok != ValueName {}) {
			//not a possible subcommand name, or no more subcommand
			return createResult(command_tok);
		}

		const auto& next_command = commandPath.back()->command();
		//now it might be a subcommand, or a positional argument
		//search for a subcommand with the same name
		const auto it = std::find_if(next_command.begin(), next_command.end(),
			[&next = **command_tok](const auto* const cmd) { return cmd->isSubcommand() && next == cmd->Name; });
		if (it == next_command.end()) {
			//not found, it is a positional argument
			return createResult(command_tok);
		}

		//found, keep descending the hierarchy
		commandPath.push_back(*it);
		lexer.expect<CommandSeparator>();
	}
}

/**
 * @brief Build a table of options in a subcommand.
 * @param command The current command/subcommand.
 * @return The option table.
*/
static pair<STPOptionTable, STPPositionalOptionTable> buildOptionTable(const STPInternal::STPBaseCommand& command) {
	STPOptionTable table;
	STPPositionalOptionTable table_positional;
	queue<const STPInternal::STPBaseCommand*> groupQueue;

	//the top-level command returned from the findCommand function must be a subcommand
	assert(command.isSubcommand());
	//we basically want to merge all options, including those in the group into one single data structure
	groupQueue.push(&command);
	while (!groupQueue.empty()) {
		const STPInternal::STPBaseCommand& current_group = *groupQueue.front();
		groupQueue.pop();

		//add non-empty option name
		const auto& member_option = current_group.option();
		for_each(member_option.begin(), member_option.end(), [&table, &table_positional](const auto* opt) {
			applyOptionName(*opt, [&table, opt](const bool is_short, const auto& opt_name) { table.try_emplace(make_pair(is_short, opt_name), opt); });
			//add positional option separately
			if (opt->isPositional()) {
				table_positional.push_back(opt);
			}
		});

		//trace child groups
		const auto& member_command = current_group.command();
		for_each(member_command.begin(), member_command.end(), [&groupQueue](const auto* cmd) {
			if (cmd->isGroup()) {
				groupQueue.push(cmd);
			}
		});
	}
	//sort position argument based on the precedence; if any 2 precedences are the same, maintain original order
	std::stable_sort(table_positional.begin(), table_positional.end(),
		[](const auto* const a, const auto* const b) { return a->PositionalPrecedence < b->PositionalPrecedence; });

	using std::move;
	return make_pair(move(table), move(table_positional));
}

/**
 * @brief Split the argument into an array of arguments using delimiter.
 * @param argument The single argument to be split.
 * @param option The option configuration, specifically the delimiter setting.
 * @param output The output array of separated arguments.
*/
static void splitArgument(string_view argument, const STPInternal::STPBaseOption& option,
	STPInternal::STPBaseOption::STPReceivedArgument& output) {
	const char delimiter = option.Delimiter;

	size_t pos;
	while (pos = argument.find(delimiter), pos != string_view::npos) {
		output.push_back(argument.substr(0u, pos));
		//plus 1 to remove the delimiter itself
		argument.remove_prefix(pos + 1u);
	}
	//store the last segment
	output.push_back(argument);
}

/**
 * @brief Read the next option.
 * @param option_type_name Specifies if the option to read is a short option, and the name of the next option.
 * @param defined_option All valid options defined by the rule.
 * @return The option parsing result.
*/
static const STPInternal::STPBaseOption& readOption(const STPOptionTypeName& option_type_name, const STPOptionTable& defined_option) {
	const auto& [is_short, option_name] = option_type_name;

	const auto it = defined_option.find(make_pair(is_short, option_name));
	if (it == defined_option.cend()) {
		ostringstream err;
		err << (is_short ? "Short" : "Long") << " option \'" << option_name << "\' is undefined" << endl;
		throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "unknown option");
	}

	//see if this option uses delimiter separated value
	const STPInternal::STPBaseOption& current_option = *it->second;
	if (current_option.used()) {
		ostringstream err;
		err << "Option \'" << option_name << "\' has already been specified, duplicate option is prohibited" << endl;
		throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "duplicate option");
	}

	return current_option;
}

/**
 * @brief Save the current state into the options, with validation.
 * @param option The current option setting.
 * @param argument The array of arguments ready to be stored into the option.
*/
static void saveOption(const STPInternal::STPBaseOption& option, const STPInternal::STPBaseOption::STPReceivedArgument& argument) {
	const auto [min, max] = option.ArgumentCount;

	//verify if number of argument meets the requirement
	if (const size_t count = argument.size();
		count < min || count > max) {
		ostringstream err;
		err << "Option \'";
		displayOption(err, option) << "\' expects minimum of " << min << " argument(s) and maximum of " << max
			<< " argument(s), but " << count << " argument(s) was/were encountered" << endl;
		throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "incorrect argument count");
	}

	//okay, convert input to desired value
	option.convert(argument);
	//if there is no problem, mark this option as parsed
	option.Result.IsUsed = true;
}

/**
 * @brief Read the next value.
 * @param current_option The current option state.
 * If current option state is null, it indicates there is no current option, and so values are treated as positional.
 * @param current_value The current value string.
 * @param positional The storage for position arguments.
 * @param argument The storage for the option argument.
 * @return True if the current option state should be reset, i.e., leaving the current state.
*/
static bool readValue(const STPOptionResult& current_option, STPStringViewAdaptor current_value,
	STPReceivedPositional& positional, STPInternal::STPBaseOption::STPReceivedArgument& argument) {
	//make some escaping
	if (!current_value->empty() && current_value->front() == '\\') {
		current_value->remove_prefix(1u);
	}

	if (const bool isPositional = !current_option;
		isPositional || argument.size() >= current_option->first->ArgumentCount.Max) {
		//no currently active option, or we got enough maximum number of argument for this option
		//then halt and treat it as a positional argument
		positional.push(current_value);
		//do not reset the state if it was a positional argument
		//because we need to save the previous state first (after this function returns),
		//and there is no state to be saved for a positional
		return !isPositional;
	}

	const auto [option_setting, use_dsv] = *current_option;
	//otherwise it is an option argument
	if (use_dsv) {
		splitArgument(*current_value, *option_setting, argument);
		//after finishing a DSV option, we expect no more arguments for this option
		//so reset current active option state
		return true;
	}
	//arguments are spread out
	argument.push_back(current_value);
	return false;
}

/**
 * @brief Read all options and values for the current command.
 * @param lexer The lexer.
 * @param current_tok An extra token consumed by the previous operation. This will be the starting token used by the command line reader.
 * @param registered_option The registered options in the current command.
 * @return All positional arguments recognised from the input.
*/
static STPReceivedPositional readCommand(STPCmdLexer& lexer, STPCmdLexer::STPToken current_tok, const STPOptionTable& registered_option) {
	//all positional arguments from the input
	STPReceivedPositional readPositional;
	//argument for each option; cleared after each option is parsed
	STPInternal::STPBaseOption::STPReceivedArgument readArgument;

	STPOptionResult current_option;
	//undefined behaviour if the current option state is empty
	const auto fastSaveOption = [&current_option, &readArgument]() -> void {
		saveOption(*current_option->first, readArgument);
	};
	const auto readOneOption = [&current_option, &fastSaveOption, &readArgument, &lexer, &registered_option]
		(const auto& expect_end_option, const STPOptionTypeName& nextOptionTypeName) -> auto {
		const string_view& nextOptionName = nextOptionTypeName.second;

		//save the previous option, if we were reading an option previously
		if (current_option) {
			fastSaveOption();
		}
		//delete old values and prepare for the next option
		readArgument.clear();

		//enter the next option state
		const STPInternal::STPBaseOption& next_option = readOption(nextOptionTypeName, registered_option);
		//check if short options are specified together, or it is the end of option statement
		const STPCmdLexer::STPToken end_option_token = expect_end_option();
		//is delimiter separated value
		const bool isDSV = end_option_token == OptionEndCompact {};
		if (isDSV && !next_option.supportDelimiter()) {
			ostringstream err;
			err << "The specified option \'" << nextOptionName
				<< "\' does not support supplying argument using delimiter style" << endl;
			lexer.throwSyntaxError(err.str(), "unsupported expression");
		}

		//create new option state
		current_option.emplace(&next_option, isDSV);

		return end_option_token;
	};
	using std::bind;
	const auto expectNextShortOption = bind(&STPCmdLexer::expect<OptionEndCompact, OptionEndSpread, ShortOptionName>, &lexer);
	const auto expectNextLongOption = bind(&STPCmdLexer::expect<OptionEndCompact, OptionEndSpread>, &lexer);

	do {
		//start from the token provided in the function argument
		//then grab from the command line
		if (current_tok == ShortOptionControl {}) {
			STPOptionTypeName short_type_name = make_pair(true, **lexer.expect<ShortOptionName>());
			//loop to consume all short options
			while (true) {
				const STPCmdLexer::STPToken short_token = readOneOption(expectNextShortOption, short_type_name);

				if (short_token != ShortOptionName {}) {
					break;
				}
				//a new option is encountered
				short_type_name.second = **short_token;
			}
		} else if (current_tok == LongOptionControl {}) {
			//long option is rather straight-forward
			const string_view long_name = **lexer.expect<LongOptionName>();
			readOneOption(expectNextLongOption, make_pair(false, long_name));
		} else if (current_tok == ValueName {}) {
			//option value, or positional argument if we are not currently parsing any option
			if (readValue(current_option, *current_tok, readPositional, readArgument)) {
				//save previous state
				assert(current_option);
				fastSaveOption();
				//leave parsing the current option
				current_option.reset();
			}
			lexer.expect<CommandSeparator>();
		} else {
			//end of command line input
			assert(current_tok == STPLexical::EndOfSequence {});
			break;
		}
		
		//grab the next option from the stream as usual
		current_tok = expectCommandSectionStart(lexer);
	} while (true);
	
	//save remaining arguments from the last option (if any)
	if (current_option) {
		fastSaveOption();
	}
	return readPositional;
}

/**
 * @brief Allocate received positional arguments to each position options in the current command.
 * @param positional All parsed positional arguments.
 * @param registered_positional All positional options in the parser rule, must be sorted based on the order of the option.
*/
static void allocatePositional(STPReceivedPositional& positional, const STPPositionalOptionTable& registered_positional) {
	STPInternal::STPBaseOption::STPReceivedArgument readArgument;
	readArgument.reserve(positional.size());
	
	//don't worry about if the number of argument allocated to an option satisfies the requirement
	//we leave this till post validation
	for (const auto* const pos_opt_ptr : registered_positional) {
		if (positional.empty()) {
			//nothing left to be allocated
			return;
		}

		const STPInternal::STPBaseOption& pos_opt = *pos_opt_ptr;
		assert(pos_opt.isPositional());
		if (pos_opt.used()) {
			//positional option might be specified as a non-positional option
			//if it is used already, skip
			continue;
		}

		const size_t remain_positional = positional.size();
		const auto [min, rule_max] = pos_opt.ArgumentCount;
		if (remain_positional < min) {
			//cannot be allocated because there is not enough argument
			break;
		}

		//the maximum possible number of argument we can/should allocated
		const size_t allocationCount = pos_opt.PositionalGreedy ? std::min(remain_positional, rule_max) : min;
		readArgument.clear();
		for (size_t i = 0u; i < allocationCount; i++) {
			readArgument.push_back(positional.front());
			positional.pop();
		}

		pos_opt.convert(readArgument);
		pos_opt.Result.IsUsed = true;
	}

	if (!positional.empty()) {
		ostringstream err;
		err << "The following option(s):" << endl;
		while (!positional.empty()) {
			err << '\t' << *positional.front() << endl;
			positional.pop();
		}
		err << "do not have any valid option to be assigned, neither they belong to any positional argument" << endl;
		
		throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "invalid argument");
	}
}

/**
 * @brief Validate if the arguments parsed satisfy the requirement.
 * Otherwise generate exception.
 * @param root The root command selected for parsing and to be checked.
 * @return A result specifies if there is any command parsed from the input.
*/
static bool postValidation(const STPInternal::STPBaseCommand& root) {
	//we will do a post-order general tree traversal; reverse BFS also works but require a stack and a queue
	//the first pointer records the current node, the second record the iterator to the next child group to be visited
	stack<pair<const STPInternal::STPBaseCommand*, const STPInternal::STPBaseCommand* const*>> groupStack;
	//result passed from child groups
	stack<bool> childGroupResult;

	//remember we should record the iterator of the pointer to child group, not the pointer itself,
	//because the array might be empty
	groupStack.emplace(&root, root.command().begin());
	while (!groupStack.empty()) {
		auto& [currentGroup, nextChild_it] = groupStack.top();

		//if there are unvisited branch left for the current node, go to that branch
		if (nextChild_it != currentGroup->command().end()) {
			const STPInternal::STPBaseCommand* const nextChild = *nextChild_it;

			if (nextChild->isGroup()) {
				//ignore child subcommand, since we are not parsing them, they are independent to the current hierarchy
				groupStack.emplace(nextChild, nextChild->command().begin());
			}
			nextChild_it++;
			continue;
		}
		//if there is no more unvisited branch, do validation then backtrack
		groupStack.pop();
		
		size_t usedOption = 0u;
		//validate options in the current group
		for (const auto* const opt : currentGroup->option()) {
			const bool used = opt->used();

			//only invalid if it is a required option but not provided
			if (!used && opt->Require) {
				ostringstream err;
				err << "Option \'";
				displayOption(err, *opt) << "\' is required but not provided in the command line" << endl;
				throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "option not provided");
			}
			if (used) {
				usedOption++;
			}
		}
		//group validation
		//it can either means there is no child group, or all child groups have done validation and we can fetch their result
		const auto& current_group_child = currentGroup->command();
		const size_t child_count = count_if(current_group_child.begin(), current_group_child.end(),
			[](const auto* const cmd) { return cmd->isGroup(); });
		for (size_t i = 0u; i < child_count; i++) {
			if (childGroupResult.top()) {
				//each used group counts as 1
				usedOption++;
			}
			childGroupResult.pop();
		}

		//summary and store result for the current group
		if (const auto [min, max] = currentGroup->OptionCount;
			usedOption < min || usedOption > max) {
			ostringstream err;
			err << "Group \'" << currentGroup->Name << "\' expects minimum of " << min << " active option(s) and maximum of "
				<< max << " active option(s), but " << usedOption << " option(s) was/were encountered" << endl;
			throw CMD_PARSER_SEMANTIC_ERROR(err.str(), "incorrect active option count");
		}

		//push the result depends on the usage of the group
		childGroupResult.push(usedOption > 0u);
	}

	//there should be left with the result from the root
	assert(childGroupResult.size() == 1u);
	return childGroupResult.top();
}

STPCommandLineParser::STPParseResult STPCommandLineParser::parse(const string& arg_encoded,
	const string_view& app_name, const STPInternal::STPBaseCommand& command) {
	constexpr static STPLexical::STPBehaviour CmdBehaviour { ArgumentDelimiter };
	STPCmdLexer lexer(arg_encoded, CmdParserName, app_name, CmdBehaviour);

	//consume the application name first
	lexer.LexicalState = CmdValueState {};
	const string_view appName = **lexer.expect<ValueName>();
	lexer.expect<CommandSeparator>();

	//locate which sub-tree/subcommand we are working on
	auto [commandPath, startingToken] = findCommand(lexer, command);
	const STPInternal::STPBaseCommand& root = *commandPath.back();
	//extract all options from this command
	const auto [optionTable, positionalTable] = buildOptionTable(root);

	//start parsing inputs in the current subcommand tree
	STPReceivedPositional parsedPositional = readCommand(lexer, startingToken, optionTable);
	//handle positional inputs at the end
	allocatePositional(parsedPositional, positionalTable);

	using std::exception_ptr;
	using std::current_exception;
	bool commandParsed = false;
	exception_ptr validationException;
	try {
		commandParsed = postValidation(root);
	} catch (...) {
		validationException = current_exception();
	}

	//validate if all parsing requirements are satisfied
	return STPCommandLineParser::STPParseResult {
		{ appName, std::move(commandPath) },
		 validationException, commandParsed
	};
}

inline static void printIndentation(ostream& stream, const streamsize indent) {
	for (streamsize i = 0; i < indent; i++) {
		stream << ' ';
	}
}

static void printNameDescription(ostream& stream, const string_view& name, const string_view& description,
	const streamsize indent, const streamsize width) {
	//start by doing indentation
	printIndentation(stream, indent);
	stream << setw(width) << std::left << name;

	//format description
	if (!description.empty()) {
		const streamsize indented_width = indent + width;
		//wrap the description to the next line if the name is too long
		//use >= to make sure at least one space is between the name and description, so it is "auto spaced"
		if (static_cast<streamsize>(name.length()) >= width) {
			stream << '\n' << setw(indented_width) << '\0';
		}
		for (const char c : description) {
			stream << c;
			if (c == '\n') {
				//wrap description and indent
				stream << setw(indented_width) << '\0';
			}
		}
	}
	stream << endl;
}

inline static void formatCountRequirement(ostream& stream, const STPInternal::STPCountRequirement& count) {
	const auto [min, max] = count;

	//min number block
	stream << '{' << min;
	if (min == max) {
		//ignore duplicate number if they are the same
		stream << '}';
		return;
	}

	//max number block
	stream << ',';
	if (count.isMaxUnlimited()) {
		stream << "...";
	} else {
		stream << max;
	}
	stream << '}';
}

template<bool Summary>
static string formatOptionName(const STPInternal::STPBaseOption& option, const char separator) {
	ostringstream stream;
	const bool isOptional = !option.Require,
		isPositional = option.isPositional();

	if constexpr (Summary) {
		if (isOptional) {
			stream << '[';
		}
		if (isPositional) {
			stream << '<';
		}
	}

	displayOption(stream, option, separator);

	if constexpr (Summary) {
		if (isPositional) {
			stream << '>';
		}
		if (isOptional) {
			stream << ']';
		}
	} else {
		if (isPositional) {
			//print positional precedence
			stream << "(:" << option.PositionalPrecedence << ')';
		}
		if (option.supportDelimiter()) {
			//tell user about the limiter if specified
			stream << '[' << option.Delimiter << ']';
		}
		formatCountRequirement(stream, option.ArgumentCount);
	}
	return stream.str();
}

template<bool Summary>
inline static string formatGroupName(const STPInternal::STPBaseCommand& group) {
	ostringstream stream;
	stream << '(' << (group.Name.empty() ? "unnamed group" : group.Name) << ')';
	if constexpr (!Summary) {
		formatCountRequirement(stream, group.OptionCount);
	}
	return stream.str();
}

static string formatSummary(const STPInternal::STPBaseCommand& root, const streamsize summary_width) {
	ostringstream stream;
	//print the content with line wrapping
	auto smartPrinter = [cumWidth = static_cast<size_t>(0u), &stream,
		width_us = static_cast<size_t>(summary_width)](const string content) mutable -> void {
		//this does not work if the content is greater than the width of the line, but we don't want to break up the content
		if (content.length() + cumWidth > width_us) {
			//line overfilled, break the line
			stream << '\n';
			cumWidth = 0u;
		}
		stream << content << ' ';
		cumWidth += content.length();
	};

	//print options first
	const auto& option = root.option();
	for_each(option.begin(), option.end(), [&smartPrinter](const auto* const opt) { smartPrinter(formatOptionName<true>(*opt, '|')); });

	//then print groups, but do not expand
	const auto& command = root.command();
	for_each(command.begin(), command.end(), [&smartPrinter](const auto* const cmd) {
		if (cmd->isGroup()) {
			smartPrinter(formatGroupName<true>(*cmd));
		}
	});

	return stream.str();
}

static void printOptionBlock(ostream& stream, const STPInternal::STPBaseCommand& group,
	const streamsize base_indent, const streamsize nested_indent, const streamsize detail_width) {
	const auto& option = group.option();
	if (option.empty()) {
		//no option in this group
		return;
	}

	printIndentation(stream, base_indent);
	stream << "Option: " << endl;
	for_each(option.begin(), option.end(), [&stream, nested_indent, width = detail_width](const auto* const opt) {
		printNameDescription(stream, formatOptionName<false>(*opt, ','), opt->Description, nested_indent, width);
	});
}

static void printGroupHeader(ostream& stream, const STPInternal::STPBaseCommand& group,
	const streamsize base_indent, const streamsize nested_indent, const streamsize summary_width) {
	//indent value is always positive
	printIndentation(stream, base_indent);
	string groupName = formatGroupName<false>(group);
	stream << "Option Group: " << groupName << endl;
	
	if (!group.Description.empty()) {
		printIndentation(stream, nested_indent);
		stream << group.Description << endl;
	}
	//print group definition
	groupName += " :=";
	//basically, we want to align the summary by the last character in the group name string
	printNameDescription(stream, groupName, formatSummary(group, summary_width), nested_indent, groupName.length() + 1u);
}

static void printGroupBlock(ostream& stream, const STPInternal::STPBaseCommand& root,
	const streamsize indent, const streamsize detail_width, const streamsize summary_width) {
	//print options in each group using DFS, also records the depth of the current node
	stack<pair<const STPInternal::STPBaseCommand*, streamsize>> groupStack;

	const auto traceGroup = [&groupStack](const STPInternal::STPBaseCommand& group, const streamsize depth) -> void {
		const auto& command = group.command();
		for_each(command.begin(), command.end(), [depth, &groupStack](const auto* const cmd) {
			if (cmd->isGroup()) {
				groupStack.emplace(cmd, depth);
			}
		});
	};
	//ignore root and go straight to the child group
	traceGroup(root, 0);

	while (!groupStack.empty()) {
		const auto [group, depth] = groupStack.top();
		groupStack.pop();
		const streamsize base_indent = indent * depth,
			group_indent = indent * (depth + 1),
			option_indent = indent * (depth + 2);

		//print group title
		printGroupHeader(stream, *group, base_indent, group_indent, summary_width);
		//print my options
		printOptionBlock(stream, *group, group_indent, option_indent, detail_width);

		//trace my child group
		traceGroup(*group, depth + 1);
	}
}

static void printSubcommandBlock(ostream& stream, const STPInternal::STPBaseCommand& root, const streamsize indent, const streamsize detail_width) {
	const auto& command = root.command();
	if (count_if(command.begin(), command.end(), [](const auto* const cmd) { return cmd->isSubcommand(); }) == 0u) {
		return;
	}

	stream << "Subcommand: " << endl;
	for_each(command.begin(), command.end(), [&stream, indent, width = detail_width](const auto* const sub) {
		if (sub->isSubcommand()) {
			//no indentation is needed for the root command
			printNameDescription(stream, sub->Name, sub->Description, indent, width);
		}
	});
}

ostream& STPCommandLineParser::operator<<(ostream& stream, const STPHelpPrinter& command_printer) {
	const auto [parse_result, indent_raw, summary_line_width_raw, detail_line_width_raw] = command_printer;
	const auto& [appName, commandPath] = *parse_result;
	//consistent with the type of the API, although I generally hate using signed type when it is not necessary...
	const streamsize indentWidth = static_cast<streamsize>(indent_raw),
		summaryLineWidth = static_cast<streamsize>(summary_line_width_raw),
		detailLineWidth = static_cast<streamsize>(detail_line_width_raw);
	//the current operating command/subcommand is the last one in the path
	const STPInternal::STPBaseCommand& workingCommand = *commandPath.back();

	//print root information
	stream << '\n' << workingCommand.Description << endl;
	//print usage line
	{
		ostringstream usage;
		usage << "Synopsis: " << appName;
		//print subcommand tree, skip the root command
		for_each(commandPath.cbegin() + 1u, commandPath.cend(), [&usage](const auto* const cmd) { usage << ' ' << cmd->Name; });
		//append a count information at the end
		formatCountRequirement(usage, workingCommand.OptionCount);

		const string usageContent = usage.str();
		//there is no indentation at the beginning of the help message
		printNameDescription(stream, usageContent, formatSummary(workingCommand, summaryLineWidth), 0, usageContent.length() + 1u);
	}
	//add an extra newline
	stream << endl;

	printOptionBlock(stream, workingCommand, 0, indentWidth, detailLineWidth);
	printGroupBlock(stream, workingCommand, indentWidth, detailLineWidth, summaryLineWidth);
	printSubcommandBlock(stream, workingCommand, indentWidth, detailLineWidth);

	return stream;
}