#include "STPCommandLineOption.h"
#include <SuperTerrain+/STPVersion.h>

//Command Line Tool
#include <SuperAlgorithm+/Parser/STPCommandLineParser.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <iostream>
#include <exception>

using std::string;
using std::make_tuple;
using std::tie;
using std::nullopt;

using std::cerr;
using std::cout;
using std::endl;

namespace Cmd = SuperTerrainPlus::STPAlgorithm::STPCommandLineParser;

using namespace STPDemo;

//The option loaded if there is no command line option supplied.
constexpr static auto DefaultOption = STPCommandLineOption::STPResult { 2.0, make_tuple(1600u, 900u), nullopt, nullopt, nullopt, nullopt };

//TODO: C++ 20: use designated initialiser to make list initialisation less verbose

/**
 * @brief Create a general help option.
 * @param useHelp A binding variable to indicate if this help option is used.
 * @return The help option.
*/
static auto createHelp(bool& useHelp) noexcept {
	useHelp = false;

	Cmd::STPOption option_help(useHelp);
	option_help.LongName = "help";
	option_help.ShortName = "h";
	option_help.Description = "Print help message to the console output and exit";

	return option_help;
}

STPCommandLineOption::STPResult STPCommandLineOption::read(const int argc, const char* const* const argv) {
	STPResult result = DefaultOption;

	bool starterHelp;
	const auto starterHelpOption = createHelp(starterHelp);

	bool printVersion = false;
	Cmd::STPOption versionOption(printVersion);
	versionOption.ShortName = "v";
	versionOption.LongName = "version";
	versionOption.Description = "Print the version information of the library and exit";

	Cmd::STPOption sprintSpeedOption(result.SprintSpeedMultiplier);
	sprintSpeedOption.LongName = "sprint-speed";
	sprintSpeedOption.Description = "Set the speed-up factor when the camera is in sprint mode";
	sprintSpeedOption.ArgumentCount.set(1u);

	Cmd::STPOption windowResolutionOption(result.WindowResolution);
	windowResolutionOption.LongName = "dim";
	windowResolutionOption.Description = "Set the rendering resolution when running windowed mode";
	windowResolutionOption.ArgumentCount.set(2u);
	windowResolutionOption.Delimiter = 'x';

	Cmd::STPOption fullScreenOption(result.UseFullScreen);
	fullScreenOption.LongName = "full-screen";
	fullScreenOption.Description = "Run the application in full-screen mode.\n"
		"Specify the index to the monitor to run,\nif unspecified, use primary monitor as default";
	fullScreenOption.ArgumentCount.Max = 1u;

	Cmd::STPOption seedOption(result.GeneratorSeed);
	seedOption.LongName = "rng-seed";
	seedOption.Description = "Set the seed value to be used by the procedural generators";
	seedOption.ArgumentCount.set(1u);

	Cmd::STPOption dayStartOption(result.DayStart);
	dayStartOption.LongName = "day-start";
	dayStartOption.Description = "Set the starting time in a day";
	dayStartOption.ArgumentCount.set(1u);

	Cmd::STPOption yearStartOption(result.YearStart);
	yearStartOption.LongName = "year-start";
	yearStartOption.Description = "Set the starting time of a year";
	yearStartOption.ArgumentCount.set(1u);

	Cmd::STPCommand resolutionGroup(tie(windowResolutionOption, fullScreenOption), tie());
	resolutionGroup.Name = "frame-resolution";
	resolutionGroup.Description = "Modifies the resolution of the rendering frame";
	resolutionGroup.OptionCount.Max = 1u;
	resolutionGroup.IsGroup = true;

	Cmd::STPCommand starterCommand(tie(
		starterHelpOption,
		versionOption,
		sprintSpeedOption,
		seedOption,
		dayStartOption,
		yearStartOption
	), tie(resolutionGroup));
	starterCommand.Name = "start";
	starterCommand.Description = "Start the SuperTerrain+ demo application by running the real-time rendering of generated terrain";
	starterCommand.OptionCount.unlimitedMax();
	
	/* -------------------------------- parsing ---------------------------------- */
#ifndef NDEBUG
	try {
		Cmd::validate(starterCommand);
	} catch (...) {
		cerr << "A validation error is encountered associated with the command line setting. "
			"This should not happen and must be a bug, please open an issue to resolve this." << endl;
		throw;
	}
#endif
	const string command_input = Cmd::encode(argc, argv);
	const Cmd::STPParseResult parser_output = Cmd::parse(command_input, "SuperDemo+", starterCommand);

	//handle message printing
	if (starterHelp) {
		const Cmd::STPHelpPrinter print_help { &parser_output.HelpData, 4u, 60u, 25u };
		cout << print_help;
		//safe exit
		std::exit(0);
	}
	if (printVersion) {
		namespace Ver = SuperTerrainPlus::STPVersion;

		cout << "SuperTerrain+ Library, SuperDemo+ application" << endl;
		cout << "Version " << Ver::Major << '.' << Ver::Minor << '.' << Ver::Patch << endl;
		std::exit(0);
	}

	//handle exception
	if (const std::exception_ptr& validation_error = parser_output.ValidationStatus;
		validation_error) {
		std::rethrow_exception(validation_error);
	}

	/* ------------------------------- numeric validation ---------------------------- */
	STP_ASSERTION_NUMERIC_DOMAIN(result.SprintSpeedMultiplier > 0.0, "The sprint speed must be strictly positive");

	const auto [resX, resY] = result.WindowResolution;
	STP_ASSERTION_NUMERIC_DOMAIN(resX > 0u && resY > 0u, "The window resolution must be strictly positive");
	if (fullScreenOption.used() && !result.UseFullScreen) {
		//default to index to the primary monitor
		result.UseFullScreen.emplace(0u);
	}

	return result;
}