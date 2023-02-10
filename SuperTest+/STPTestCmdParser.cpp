//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

//Parser
#include <SuperAlgorithm+/Parser/Framework/STPBasicStringAdaptor.h>
#include <SuperAlgorithm+/Parser/STPCommandLineParser.h>

//Error
#include <SuperTerrain+/Exception/STPValidationFailed.h>
#include <SuperTerrain+/Exception/STPParserError.h>

#include <string>
#include <string_view>
#include <sstream>
#include <limits>
#include <algorithm>
#include <exception>

//Storage
#include <array>
#include <vector>
#include <tuple>

using std::string;
using std::string_view;
using std::ostringstream;
using std::array;
using std::vector;
using std::tuple;

using std::tie;
using std::make_tuple;

using Catch::Matchers::ContainsSubstring;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPAlgorithm;
namespace Cmd = STPCommandLineParser;

#define THROW_AS_VALIDATION_ERROR(EXPR) REQUIRE_THROWS_AS(EXPR, STPException::STPValidationFailed)
#define THROW_AS_PARSER_ERROR(EXPR) REQUIRE_THROWS_AS(EXPR, STPException::STPParserError::STPBasic)
#define COMPARE_FLOAT(VALUE, TARGET) CHECK_THAT(VALUE, Catch::Matchers::WithinRel(TARGET, std::numeric_limits<float>::epsilon() * 500.0f))

template<template<typename> class O, typename BT>
static void bindVariable(O<BT>& option, BT& variable) {
	option.Variable = &variable;
}

SCENARIO("STPCommandLineParser user-defined data structure can perform certain operations", "[AlgorithmHost][STPCommandLineParser]") {

	GIVEN("A number of options with binding variable") {
		Cmd::STPOption<void> VoidOption;

		bool BoolValue = true;
		Cmd::STPOption<bool> BoolOption;
		BoolOption.InferredValue = false;
		bindVariable(BoolOption, BoolValue);

		float FloatValue = 0.0f;
		Cmd::STPOption<float> FloatOption;
		bindVariable(FloatOption, FloatValue);

		typedef vector<unsigned int> STPArrayOption;
		STPArrayOption ArrayValue;
		Cmd::STPOption<STPArrayOption> ArrayIntOption;
		bindVariable(ArrayIntOption, ArrayValue);

		typedef tuple<bool, float, unsigned int> STPTupleOption;
		STPTupleOption TupleValue = make_tuple(false, 0.0f, 0u);
		Cmd::STPOption<STPTupleOption> TupleOption;
		bindVariable(TupleOption, TupleValue);

		constexpr static STPStringViewAdaptor BoolString = "false", FloatString = "3.14f",
			ArrayElementValue = "666u", TupleOne = "true", TupleTwo = "-2.71f", TupleThree = "888u";
		WHEN("Valid parsed string are obtained") {

			THEN("Parsed string can be converted to corresponding binding variable") {
				CHECK_NOTHROW(VoidOption.convert({}));
				CHECK_NOTHROW(BoolOption.convert({ BoolString }));
				CHECK_NOTHROW(FloatOption.convert({ FloatString }));
				CHECK_NOTHROW(ArrayIntOption.convert({ 5u, ArrayElementValue }));
				CHECK_NOTHROW(TupleOption.convert({ TupleOne, TupleTwo, TupleThree }));

				CHECK(BoolValue == false);
				COMPARE_FLOAT(FloatValue, 3.14f);
				CHECK(std::all_of(ArrayValue.cbegin(), ArrayValue.cend(), [](const auto num) { return num == 666u; }));
				const auto [t1, t2, t3] = TupleValue;
				CHECK(t1 == true);
				COMPARE_FLOAT(t2, -2.71f);
				CHECK(t3 == 888u);
			}

		}

		WHEN("Invalid parsed string are encountered") {
			constexpr static STPStringViewAdaptor BrokenBool = "tu?re", BrokenFloat = "hello,world";

			THEN("Binding variable conversion fails") {
				//invalid format
				THROW_AS_PARSER_ERROR(BoolOption.convert({ BrokenBool }));
				THROW_AS_PARSER_ERROR(FloatOption.convert({ BrokenFloat }));

				//incorrect number of argument
				THROW_AS_VALIDATION_ERROR(BoolOption.convert({ BoolString, BoolString }));
				THROW_AS_VALIDATION_ERROR(FloatOption.convert({ }));
				THROW_AS_VALIDATION_ERROR(TupleOption.convert({ TupleOne }));
			}

		}

	}

}

class CmdParserTester {
private:

	auto createHelp(bool& use_help) noexcept {
		Cmd::STPOption<bool> OptHelp;
		OptHelp.Variable = &use_help;
		OptHelp.InferredValue = true;
		OptHelp.ShortName = "h";
		OptHelp.LongName = "help";
		OptHelp.Description = "Print help message for the current command and exit";

		return OptHelp;
	}

	void resetString() noexcept {
		constexpr static auto reset = [](string_view& s) constexpr noexcept -> void {
			s = string_view();
		};
		//just to prevent string view from holding dangling pointer
		reset(this->CloneRepoName);
		reset(this->PullOriginName);
		reset(this->PullBranchName);
		reset(this->AddOneFileName);
		this->AddAllFileName.clear();
		reset(this->CommitMessageValue);
	}

protected:

	string CommandLineSource;

	string_view CloneRepoName;
	unsigned int CloneDepthValue;

	string_view PullOriginName, PullBranchName;

	bool AddVerboseValue;
	string_view AddOneFileName;
	vector<string_view> AddAllFileName;

	string_view CommitMessageValue;

	//This command automatically manages the lifetime of the string argument
	//Provide nullptr to the help stream to disable printing help message.
	bool startParser(string&& argument, ostringstream* const help_stream = nullptr) {
		/* ------------------ clone ---------------------- */
		bool ClonePrintHelp = false;
		const auto CloneHelp = this->createHelp(ClonePrintHelp);

		Cmd::STPOption<string_view> CloneRepo;
		CloneRepo.LongName = "repo";
		CloneRepo.Description = "Clone a remote repository to local hard drive";
		CloneRepo.ArgumentCount.set(1u);
		CloneRepo.PositionalPrecedence = 1u;
		CloneRepo.Require = true;
		CloneRepo.Variable = &this->CloneRepoName;

		Cmd::STPOption<unsigned int> CloneDepth;
		CloneDepth.LongName = "depth";
		CloneDepth.Description = "Specify the number of commit to be clone, starting from HEAD";
		CloneDepth.Delimiter = ',';
		CloneDepth.ArgumentCount.set(1u);
		CloneDepth.Variable = &this->CloneDepthValue;

		Cmd::STPCommand CloneCommand(tie(CloneHelp, CloneRepo, CloneDepth), tie());
		CloneCommand.Name = "clone";
		CloneCommand.Description = "Clone repository";
		CloneCommand.OptionCount.Min = 1u;
		CloneCommand.OptionCount.Max = 2u;

		/* ------------------- pull ----------------------- */
		bool PullPrintHelp = false;
		const auto PullHelp = this->createHelp(PullPrintHelp);

		Cmd::STPOption<string_view> PullOrigin;
		PullOrigin.ShortName = "o";
		PullOrigin.Description = "Fetch latest commits from the origin";
		PullOrigin.ArgumentCount.set(1u);
		PullOrigin.PositionalPrecedence = 1u;
		PullOrigin.Require = true;
		PullOrigin.Variable = &this->PullOriginName;

		auto PullBranch = PullOrigin;
		PullBranch.ShortName = "b";
		PullBranch.Description = "Merge pulled comfits into the target branch";
		PullBranch.PositionalPrecedence = 2u;
		PullBranch.Variable = &this->PullBranchName;

		Cmd::STPCommand PullOriginBranchGroup(tie(PullOrigin, PullBranch), tie());
		PullOriginBranchGroup.Name = "pull-origin-branch";
		PullOriginBranchGroup.Description = "Specify the origin where commits are pulled\nand branch to merge into";
		PullOriginBranchGroup.OptionCount.set(2u);
		PullOriginBranchGroup.IsGroup = true;

		Cmd::STPCommand PullCommand(tie(PullHelp), tie(PullOriginBranchGroup));
		PullCommand.Name = "pull";
		PullCommand.Description = "Pull repository";
		PullCommand.OptionCount.set(1u);

		/* ------------------ add ------------------------ */
		bool AddPrintHelp = false;
		const auto AddHelp = this->createHelp(AddPrintHelp);

		Cmd::STPOption<void> AddVerbose;
		AddVerbose.ShortName = "v";
		AddVerbose.LongName = "verbose";
		AddVerbose.Description = "Show diagnostic information when executing add command";

		Cmd::STPOption<string_view> AddOneFile;
		AddOneFile.LongName = "file";
		AddOneFile.ShortName = "f";
		AddOneFile.Description = "Stage changes for one file";
		AddOneFile.ArgumentCount.set(1u);
		AddOneFile.Variable = &this->AddOneFileName;

		Cmd::STPOption<vector<string_view>> AddAllFile;
		AddAllFile.LongName = "all-file";
		AddAllFile.Description = "Add a list of all files to staged changes";
		AddAllFile.Delimiter = '|';
		AddAllFile.ArgumentCount.Min = 1u;
		AddAllFile.ArgumentCount.unlimitedMax();
		AddAllFile.PositionalPrecedence = 1u;
		AddAllFile.PositionalGreedy = true;
		AddAllFile.Variable = &this->AddAllFileName;

		Cmd::STPCommand AddFileGroup(tie(AddOneFile, AddAllFile), tie());
		AddFileGroup.Name = "add-file";
		AddFileGroup.Description = "Add one, or a number of file to staged changes";
		AddFileGroup.OptionCount.set(1u);
		AddFileGroup.IsGroup = true;

		Cmd::STPCommand AddCommand(tie(AddHelp, AddVerbose), tie(AddFileGroup));
		AddCommand.Name = "add";
		AddCommand.Description = "Add working files";
		AddCommand.OptionCount.Max = 2u;

		/* ----------------- commit ----------------------- */
		bool CommitPrintHelp = false;
		const auto CommitHelp = this->createHelp(CommitPrintHelp);

		Cmd::STPOption<string_view> CommitMessage;
		CommitMessage.ShortName = "m";
		CommitMessage.Description = "Add optional message to be displayed in the commit log";
		CommitMessage.ArgumentCount.set(1u);
		CommitMessage.Variable = &this->CommitMessageValue;

		Cmd::STPCommand CommitCommand(tie(CommitHelp, CommitMessage), tie());
		CommitCommand.Name = "commit";
		CommitCommand.Description = "Commit changes";
		CommitCommand.OptionCount.Max = 1u;

		/* ------------- starting command ---------------- */
		bool GitPrintHelp = false;
		const auto GitHelp = this->createHelp(GitPrintHelp);

		Cmd::STPCommand MiniGit(tie(GitHelp), tie(CloneCommand, PullCommand, AddCommand, CommitCommand));
		MiniGit.Name = "mini-git";
		MiniGit.Description = "A test program for SuperTerrain+ command line parser using GIT style command line";
		MiniGit.OptionCount.set(1u);

		/* ------------------ execution ---------------- */
		REQUIRE_NOTHROW(Cmd::validate(MiniGit));

		this->resetString();
		//copy the sourcing encoded command line data
		this->CommandLineSource = std::move(argument);
		const auto ParseResult = Cmd::parse(this->CommandLineSource, "SuperTerrain+ Command Line Parser Test", MiniGit);
		//inferred bool value based on if the option is used
		this->AddVerboseValue = AddVerbose.Result.IsUsed;
		
		//ignore validation error if we need to print help and exit
		if (help_stream && (ClonePrintHelp || PullPrintHelp || AddPrintHelp || CommitPrintHelp || GitPrintHelp)) {
			const Cmd::STPHelpPrinter Printer { &ParseResult.HelpData, 4u, 40u };
			*help_stream << Printer;
		} else if (const auto& Validation = ParseResult.ValidationStatus;
			Validation) {
			//if we don't print help, need to check for validation
			std::rethrow_exception(Validation);
		}

		return ParseResult.NonEmptyCommandLine;
	}

public:

	inline CmdParserTester() noexcept : CloneDepthValue(std::numeric_limits<unsigned int>::max()), AddVerboseValue(false) {

	}

};

template<typename... C>
inline static string getEncodedArgument(const C* const... arg) {
	constexpr static char AppName[] = "./CmdParserTest";

	const array Argument { AppName, arg... };
	return Cmd::encode(static_cast<int>(Argument.size()), Argument.data());
}

#define RUN_VALIDATION_MATCH(INPUT, MAT) CHECK_THROWS_WITH(Cmd::validate(INPUT), MAT)

#define RUN_PARSER(INPUT) CHECK(this->startParser(INPUT))

#define RUN_PARSER_ERROR(INPUT) THROW_AS_PARSER_ERROR(this->startParser(INPUT))
#define RUN_PARSER_ERROR_MATCH(INPUT, MAT) CHECK_THROWS_WITH(this->startParser(INPUT), MAT)

SCENARIO_METHOD(CmdParserTester, "STPCommandLineParser can parsed command line based on application configuration",
	"[AlgorithmHost][STPCommandLineParser]") {

	GIVEN("An ill-formed application") {
		bool FlagValue = false;
		Cmd::STPOption<bool> Flag;
		Flag.ShortName = "f";
		Flag.Variable = &FlagValue;

		unsigned int IntValue = 0u;
		Cmd::STPOption<unsigned int> IntOption;
		IntOption.LongName = "data";
		IntOption.Variable = &IntValue;
		IntOption.ArgumentCount.set(1u);

		Cmd::STPCommand Group(tie(Flag), tie());
		Group.Name = "test-group";
		Group.IsGroup = true;

		Cmd::STPCommand Root(tie(IntOption), tie(Group));
		Root.Name = "test-command";

		WHEN("The parser is asked to validate the application given their underlying problem") {
			//invariant for fool proof
			REQUIRE_NOTHROW(Cmd::validate(Root));

			//split tests into different branches so every time the preconditions are re-initialised
			THEN("Root cannot be a group") {
				RUN_VALIDATION_MATCH(Group, ContainsSubstring("not be a group"));
			}

			THEN("Name of a (sub)command or group must not be empty") {
				Root.Name = string_view();
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("Name") && ContainsSubstring("non-empty"));
			}

			THEN("Option must have at least one non-empty name") {
				Flag.ShortName = string_view();
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("option name"));
			}

			THEN("The format of the option name must be valid") {
				IntOption.LongName = "?";
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("format") && ContainsSubstring("invalid"));
			}

			THEN("The range of count must be reasonable") {
				Flag.ArgumentCount.Min = 123u;
				RUN_VALIDATION_MATCH(Root,
					ContainsSubstring("minimum") && ContainsSubstring("greater than") && ContainsSubstring("maximum"));
			}

			THEN("Duplicate option cannot appear in the same subcommand hierarchy") {
				Cmd::STPCommand DuplicateGroup(tie(Flag, IntOption), tie());
				DuplicateGroup.Name = "duplicate-group";
				DuplicateGroup.IsGroup = true;

				Cmd::STPCommand DuplicateRoot(tie(IntOption), tie(DuplicateGroup));
				DuplicateRoot.Name = "duplicate-root";

				RUN_VALIDATION_MATCH(DuplicateRoot, ContainsSubstring("data") && ContainsSubstring("not unique"));
			}

			THEN("Subcommand name must be unique in the same hierarchy") {
				Cmd::STPCommand SubA(tie(Flag), tie());
				SubA.Name = "dummy-name";
				Cmd::STPCommand SubB(tie(IntOption), tie());
				SubB.Name = SubA.Name;

				Cmd::STPCommand AnotherRoot(tie(), tie(SubA, SubB));
				AnotherRoot.Name = "another-root";

				RUN_VALIDATION_MATCH(AnotherRoot, ContainsSubstring("dummy-name") && ContainsSubstring("redefined"));
			}

			THEN("A group must not contain any subcommand in its hierarchy") {
				Cmd::STPCommand Sub(tie(Flag), tie());
				Sub.Name = "just-a-subcommand";

				Cmd::STPCommand GroupA(tie(IntOption), tie(Sub));
				GroupA.Name = "groupA";
				GroupA.IsGroup = true;
				Cmd::STPCommand GroupB(tie(), tie(GroupA));
				GroupB.Name = "groupB";
				GroupB.IsGroup = true;

				Cmd::STPCommand AnotherRoot(tie(), tie(GroupB));
				AnotherRoot.Name = "another-root";

				RUN_VALIDATION_MATCH(AnotherRoot, ContainsSubstring("groupA") && ContainsSubstring("just-a-subcommand"));
			}

		}

	}

	GIVEN("A well-formed application") {

		WHEN("Application receives some command line arguments with correct syntax") {
			constexpr static char CommitMessage[] = "Fix some fatal errors",
				OneFileName[] = "./input.txt", CloneRepo[] = "https://www.example.com";

			THEN("The parsing result can be verified") {
				const auto Trial = GENERATE(range(0u, 5u));
				switch (Trial) {
				case 0u:
					RUN_PARSER(getEncodedArgument("commit", "-m", CommitMessage));
					CHECK((this->CommitMessageValue == CommitMessage));
					break;
				case 1u:
					RUN_PARSER(getEncodedArgument("add", "-vf", OneFileName));
					CHECK(this->AddVerboseValue);
					CHECK((this->AddOneFileName == OneFileName));
					break;
				case 2u:
					RUN_PARSER(getEncodedArgument("add", "--all-file=./source.h|./source.cpp|./source.inl"));
					CHECK(this->AddAllFileName.size() == 3u);
					CHECK((this->AddAllFileName[1] == "./source.cpp"));
					CHECK_FALSE(this->AddVerboseValue);
					break;
				case 3u:
					RUN_PARSER(getEncodedArgument("pull", "origin", "master"));
					CHECK((this->PullOriginName == "origin"));
					CHECK((this->PullBranchName == "master"));
					break;
				case 4u:
					RUN_PARSER(getEncodedArgument("clone", "--depth=5", CloneRepo));
					CHECK(this->CloneDepthValue == 5u);
					CHECK((this->CloneRepoName == CloneRepo));
					break;
				default:
					break;
				}
			}

		}

		WHEN("The application receives erroneous command line arguments") {

			THEN("Parser errors are generated") {
				//subcommand is interpreted as positional argument
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("destroy", "--all"),
					ContainsSubstring("all") && ContainsSubstring("undefined"));
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("commit", "-S"),
					ContainsSubstring("S") && ContainsSubstring("undefined"));
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("pull", "nowhere"),
					ContainsSubstring("b") && ContainsSubstring("not provided"));
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("clone", "--depth=1,2,3"),
					ContainsSubstring("maximum of 1"));
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("add", "--file", "fileA.txt", "--all-file", "fileB.txt"),
					ContainsSubstring("maximum of 1"));
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("add", "-v", "-v"),
					ContainsSubstring("v") && ContainsSubstring("duplicate"));
				RUN_PARSER_ERROR(getEncodedArgument("clone", "--depth", "abc"));
			}

		}

	}

}