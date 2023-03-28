//Catch2
#include <catch2/catch_test_macros.hpp>
//Matcher
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
//Generator
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

//Parser
#include <SuperAlgorithm+Host/Parser/Framework/STPBasicStringAdaptor.h>
#include <SuperAlgorithm+Host/Parser/STPCommandLineParser.h>

//Error
#include <SuperTerrain+/Exception/STPValidationFailed.h>
#include <SuperTerrain+/Exception/STPParserError.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <string>
#include <sstream>
#include <limits>
#include <exception>
#include <algorithm>

//Storage
#include <array>
#include <vector>
#include <tuple>
#include <optional>

using std::string;
using std::ostringstream;
using std::array;
using std::vector;
using std::tuple;
using std::optional;

using std::tie;
using std::make_tuple;

using Catch::Matchers::SizeIs;
using Catch::Matchers::IsEmpty;
using Catch::Matchers::Equals;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::AllMatch;
using Catch::Matchers::Predicate;
using Catch::Matchers::MessageMatches;

namespace Err = SuperTerrainPlus::STPException;
namespace Cmd = SuperTerrainPlus::STPAlgorithm::STPCommandLineParser;

#define THROW_AS_PARSER_ERROR(EXPR) REQUIRE_THROWS_AS(EXPR, Err::STPParserError::STPSemanticError)
#define COMPARE_FLOAT(VALUE, TARGET) CHECK_THAT(VALUE, Catch::Matchers::WithinRel(TARGET, std::numeric_limits<float>::epsilon() * 500.0f))
#define COMPARE_FLOAT_RESULT(VALUE, EXPECTED) CHECK_THAT(VALUE, Catch::Matchers::WithinULP(EXPECTED, 50ull))

SCENARIO("STPCommandLineParser user-defined data structure can perform certain operations", "[AlgorithmHost][STPCommandLineParser]") {

	GIVEN("Some random options and commands to be queried with their auxiliary functions") {
		unsigned int IntData = 0u;
		Cmd::STPOption Option(IntData);

		Cmd::STPCommand Command(tie(Option), tie());

		WHEN("Argument count is modified through helper functions") {
			auto& OptionCount = Option.ArgumentCount;
			auto& CommandCount = Command.OptionCount;

			OptionCount.set(123u);
			CommandCount.unlimitedMax();

			THEN("The count variables are modified correctly") {
				CHECK(OptionCount.Min == OptionCount.Max);
				CHECK(OptionCount.Min == 123u);

				CHECK(CommandCount.isMaxUnlimited());
			}

		}

		WHEN("Option and command settings are accessed via their functions") {
			
			THEN("These functions return the default value") {
				CHECK_FALSE(Option.isPositional());
				CHECK_FALSE(Option.supportDelimiter());
				CHECK_FALSE(Option.used());

				CHECK_FALSE(Command.isGroup());
				CHECK(Command.isSubcommand());
			}

			AND_WHEN("Their settings are modified") {
				Option.PositionalPrecedence = 888u;
				Option.Delimiter = '?';
				Option.Result.IsUsed = true;

				Command.IsGroup = true;

				THEN("The functions return value corresponds to their current state") {
					CHECK(Option.isPositional());
					CHECK(Option.supportDelimiter());
					CHECK(Option.used());

					CHECK(Command.isGroup());
					CHECK_FALSE(Command.isSubcommand());
				}

			}

		}

		WHEN("Command tree is requested") {
			const auto& OptionTree = Command.option();
			const auto& CommandTree = Command.command();

			THEN("Tree container can be accessed through its functions") {
				CHECK_THAT(OptionTree, SizeIs(1u));
				CHECK_THAT(CommandTree, IsEmpty());

				using std::distance;
				CHECK(distance(OptionTree.begin(), OptionTree.end()) == 1u);
				CHECK(distance(CommandTree.begin(), CommandTree.end()) == 0u);
			}

		}

	}

	GIVEN("A number of options with binding variable") {

		WHEN("Binding variables are simple types with single level of depth") {
			Cmd::STPOption<void> VoidOption;

			bool BoolValue = true;
			Cmd::STPOption BoolOption(BoolValue);
			BoolOption.InferredValue = false;

			float FloatValue = 0.0f;
			Cmd::STPOption FloatOption(FloatValue);

			vector<unsigned int> ArrayValue;
			Cmd::STPOption ArrayIntOption(ArrayValue);

			tuple<bool, float, unsigned int> TupleValue = make_tuple(false, 0.0f, 0u);
			Cmd::STPOption TupleOption(TupleValue);

			optional<unsigned int> OptionalValue;
			Cmd::STPOption OptionalOption(OptionalValue);

			THEN("Parsed string can be converted to corresponding binding variable") {
				CHECK_NOTHROW(VoidOption.convert({}));
				CHECK_NOTHROW(BoolOption.convert({ "false" }));
				CHECK_NOTHROW(FloatOption.convert({ "3.14f" }));
				CHECK_NOTHROW(ArrayIntOption.convert({ 5u, "666u" }));
				CHECK_NOTHROW(TupleOption.convert({ "true", "-2.71f", "888u" }));
				CHECK_NOTHROW(OptionalOption.convert({ "12345u" }));

				//-----------
				//value check
				CHECK(BoolValue == false);
				COMPARE_FLOAT(FloatValue, 3.14f);
				CHECK_THAT(ArrayValue, AllMatch(Predicate<unsigned int>([](const auto num) { return num == 666u; })));

				const auto [t1, t2, t3] = TupleValue;
				CHECK(t1 == true);
				COMPARE_FLOAT(t2, -2.71f);
				CHECK(t3 == 888u);

				REQUIRE(OptionalValue);
				CHECK(*OptionalValue == 12345u);

				//-----------
				//a separate test case for optional in case of empty argument
				OptionalValue.reset();
				CHECK_NOTHROW(OptionalOption.convert({}));
				CHECK_FALSE(OptionalValue);
			}

			AND_WHEN("Invalid parsed string are encountered") {

				THEN("Binding variable conversion fails") {
					//invalid format
					THROW_AS_PARSER_ERROR(BoolOption.convert({ "tu?re" }));
					THROW_AS_PARSER_ERROR(FloatOption.convert({ "hello,world" }));

					//incorrect number of argument
					THROW_AS_PARSER_ERROR(BoolOption.convert({ "true", "false" }));
					THROW_AS_PARSER_ERROR(TupleOption.convert({ "false" }));
				}

			}

		}

		WHEN("Binding variables are more complex with nested definitions") {
			vector<tuple<float, float>> ArrayTwoFloatValue;
			Cmd::STPOption ArrayTwoFloatOption(ArrayTwoFloatValue);

			optional<tuple<unsigned int, bool>> OptionalTupleValue;
			Cmd::STPOption OptionalTupleOption(OptionalTupleValue);

			tuple<tuple<float, int>, float> ComplexTupleValue;
			Cmd::STPOption ComplexTupleOption(ComplexTupleValue);

			THEN("All nested types can be converted") {
				constexpr static float Avogadro = 6.02f;

				CHECK_NOTHROW(ArrayTwoFloatOption.convert({ "2.5f", "15.05f", "6.0f", "36.12f" }));
				CHECK_NOTHROW(OptionalTupleOption.convert({ "1945u", "true" }));//the year ENIAC was invented
				CHECK_NOTHROW(ComplexTupleOption.convert({ "6.625f", "-34", "6.28f" }));//Planck's constant

				//-----------
				//value check
				for (const auto [a, result] : ArrayTwoFloatValue) {
					COMPARE_FLOAT_RESULT(a * Avogadro, result);
				}
				REQUIRE(OptionalTupleValue);

				const auto [t1, t2] = *OptionalTupleValue;
				CHECK(t1 == 1945u);
				CHECK(t2);

				const auto [t3, u1] = ComplexTupleValue;
				const auto [v1, v2] = t3;
				CHECK(v1 == 6.625f);
				CHECK(v2 == -34);
				CHECK(u1 == 6.28f);
			}

			AND_WHEN("Nested binding variables are broken") {

				THEN("Conversion fails immediately regardless of the nesting depth") {
					THROW_AS_PARSER_ERROR(ArrayTwoFloatOption.convert({ "-1.2f", "2.4f", "-3.6f" }));
					THROW_AS_PARSER_ERROR(OptionalTupleOption.convert({ "555u", "false", "0u" }));
					THROW_AS_PARSER_ERROR(ComplexTupleOption.convert({ "888.88f" }));
				}

			}

		}

	}

}

class CmdParserTester {
private:

	auto createHelp(bool& use_help) noexcept {
		Cmd::STPOption OptHelp(use_help);
		OptHelp.InferredValue = true;
		OptHelp.ShortName = "h";
		OptHelp.LongName = "help";
		OptHelp.Description = "Print help message for the current command and exit";

		return OptHelp;
	}

protected:

	string SubcommandBranchName;

	string CloneRepoName;
	unsigned int CloneDepthValue;

	string PullOriginName, PullBranchName;

	bool AddVerboseValue;
	string AddOneFileName;
	vector<string> AddAllFileName;

	string CommitMessageValue;

	//This command automatically manages the lifetime of the string argument
	//Provide nullptr to the help stream to disable printing help message.
	bool startParser(string argument, ostringstream* const help_stream = nullptr) {
		/* ------------------ clone ---------------------- */
		bool ClonePrintHelp = false;
		const auto CloneHelp = this->createHelp(ClonePrintHelp);

		Cmd::STPOption CloneRepo(this->CloneRepoName);
		CloneRepo.LongName = "repo";
		CloneRepo.Description = "Clone a remote repository to local hard drive";
		CloneRepo.ArgumentCount.set(1u);
		CloneRepo.PositionalPrecedence = 1u;
		CloneRepo.Require = true;

		Cmd::STPOption CloneDepth(this->CloneDepthValue);
		CloneDepth.LongName = "depth";
		CloneDepth.Description = "Specify the number of commit to be clone, starting from HEAD";
		CloneDepth.Delimiter = ',';
		CloneDepth.ArgumentCount.set(1u);

		Cmd::STPCommand CloneCommand(tie(CloneHelp, CloneRepo, CloneDepth), tie());
		CloneCommand.Name = "clone";
		CloneCommand.Description = "Clone repository";
		CloneCommand.OptionCount.Min = 1u;
		CloneCommand.OptionCount.Max = 2u;

		/* ------------------- pull ----------------------- */
		bool PullPrintHelp = false;
		const auto PullHelp = this->createHelp(PullPrintHelp);

		Cmd::STPOption PullOrigin(this->PullOriginName);
		PullOrigin.ShortName = "o";
		PullOrigin.Description = "Fetch latest commits from the origin";
		PullOrigin.ArgumentCount.set(1u);
		PullOrigin.PositionalPrecedence = 1u;
		PullOrigin.Require = true;

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

		Cmd::STPOption AddOneFile(this->AddOneFileName);
		AddOneFile.LongName = "file";
		AddOneFile.ShortName = "f";
		AddOneFile.Description = "Stage changes for one file";
		AddOneFile.ArgumentCount.set(1u);

		Cmd::STPOption AddAllFile(this->AddAllFileName);
		AddAllFile.LongName = "all-file";
		AddAllFile.Description = "Add a list of all files to staged changes";
		AddAllFile.Delimiter = '|';
		AddAllFile.ArgumentCount.Min = 1u;
		AddAllFile.ArgumentCount.unlimitedMax();
		AddAllFile.PositionalPrecedence = 1u;
		AddAllFile.PositionalGreedy = true;

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

		Cmd::STPOption CommitMessage(this->CommitMessageValue);
		CommitMessage.ShortName = "m";
		CommitMessage.Description = "Add optional message to be displayed in the commit log";
		CommitMessage.ArgumentCount.set(1u);

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

		const auto ParseResult = Cmd::parse(argument, "SuperTerrain+ Command Line Parser Test", MiniGit);
		this->SubcommandBranchName = ParseResult.commandBranch().Name;

		//inferred bool value based on if the option is used
		this->AddVerboseValue = AddVerbose.used();
		
		//ignore validation error if we need to print help and exit
		if (help_stream && (ClonePrintHelp || PullPrintHelp || AddPrintHelp || CommitPrintHelp || GitPrintHelp)) {
			const Cmd::STPHelpPrinter Printer { &ParseResult.HelpData, 4u, 60u, 40u };
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

#define RUN_VALIDATION_GENERIC(INPUT, EXC, MAT) CHECK_THROWS_MATCHES(Cmd::validate(INPUT), EXC, MessageMatches(MAT))
#define RUN_VALIDATION_MATCH(INPUT, MAT) RUN_VALIDATION_GENERIC(INPUT, Err::STPValidationFailed, MAT)
#define RUN_NUMERIC_MATCH(INPUT, MAT) RUN_VALIDATION_GENERIC(INPUT, Err::STPNumericDomainError, MAT)

#define RUN_PARSER(INPUT) CHECK(this->startParser(INPUT))

#define RUN_PARSER_ERROR(INPUT) THROW_AS_PARSER_ERROR(this->startParser(INPUT))
#define RUN_PARSER_ERROR_MATCH(INPUT, MAT) CHECK_THROWS_MATCHES(this->startParser(INPUT), \
	Err::STPParserError::STPSemanticError, MessageMatches(MAT))

SCENARIO_METHOD(CmdParserTester, "STPCommandLineParser can parsed command line based on application configuration",
	"[AlgorithmHost][STPCommandLineParser]") {

	GIVEN("An ill-formed application") {
		bool FlagValue = false;
		Cmd::STPOption Flag(FlagValue);
		Flag.ShortName = "f";

		unsigned int IntValue = 0u;
		Cmd::STPOption IntOption(IntValue);
		IntOption.LongName = "data";
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
				Root.Name = {};
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("Name") && ContainsSubstring("non-empty"));
			}

			THEN("Option must have at least one non-empty name") {
				Flag.ShortName = {};
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("option name"));
			}

			THEN("The format of the option name must be valid") {
				IntOption.LongName = "?";
				RUN_VALIDATION_MATCH(Root, ContainsSubstring("format") && ContainsSubstring("invalid"));
			}

			THEN("The range of count must be reasonable") {
				Flag.ArgumentCount.Min = 123u;
				RUN_NUMERIC_MATCH(Root,
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
					CHECK_THAT(this->SubcommandBranchName, Equals("commit"));
					CHECK_THAT(this->CommitMessageValue, Equals(CommitMessage));
					break;
				case 1u:
					RUN_PARSER(getEncodedArgument("add", "-vf", OneFileName));
					CHECK_THAT(this->SubcommandBranchName, Equals("add"));
					CHECK(this->AddVerboseValue);
					CHECK_THAT(this->AddOneFileName, Equals(OneFileName));
					break;
				case 2u:
					RUN_PARSER(getEncodedArgument("add", "--all-file=./source.h|./source.cpp|./source.inl"));
					CHECK_THAT(this->SubcommandBranchName, Equals("add"));
					CHECK_THAT(this->AddAllFileName, SizeIs(3u));
					CHECK_THAT(this->AddAllFileName[1], Equals("./source.cpp"));
					CHECK_FALSE(this->AddVerboseValue);
					break;
				case 3u:
					RUN_PARSER(getEncodedArgument("pull", "origin", "master"));
					CHECK_THAT(this->SubcommandBranchName, Equals("pull"));
					CHECK_THAT(this->PullOriginName, Equals("origin"));
					CHECK_THAT(this->PullBranchName, Equals("master"));
					break;
				case 4u:
					RUN_PARSER(getEncodedArgument("clone", "--depth=5", CloneRepo));
					CHECK_THAT(this->SubcommandBranchName, Equals("clone"));
					CHECK(this->CloneDepthValue == 5u);
					CHECK_THAT(this->CloneRepoName, Equals(CloneRepo));
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
				RUN_PARSER_ERROR_MATCH(getEncodedArgument("add", "--v"),
					ContainsSubstring("v") && ContainsSubstring("undefined"));
				RUN_PARSER_ERROR(getEncodedArgument("clone", "--depth", "abc"));
			}

		}

	}

}