#pragma once

//System
#include <iostream>
#include <sstream>
//Base reporter
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_cumulative_base.hpp>

//Emit Helper
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_console_width.hpp>
#include <catch2/catch_test_case_info.hpp>

using namespace Catch;

using std::endl;

using std::string;
using std::stringstream;

/**
 * @brief STPConsoleReporter is a test reporter for SuperTerrain+, it outputs a human-readable test report and is best suited for console output.
*/
class STPConsoleReporter : public CumulativeReporterBase {
private:

	/**
	 * @brief Emit a text to the stream that is centred at the console
	 * @param text The text to be centred
	*/
	void emitCentreString(const string& text) const {
		if (text.size() > CATCH_CONFIG_CONSOLE_WIDTH) {
			//cannot centre the tetx because it's too wide
			this->stream << text;
			return;
		}

		auto emit_border = [this](size_t border_size) -> void {
			for (size_t i = 0ull; i < border_size / 2ull; i++) {
				this->stream << ' ';
			}
		};
		//space left empty
		const size_t border = CATCH_CONFIG_CONSOLE_WIDTH - text.size();

		//emit the left border
		emit_border(border);

		//emit the text
		this->stream << text;

		//emit the right border
		emit_border(border);

		this->stream << endl;
	}

	/**
	 * @brief Emit a text that is aligned to the right border of the console
	 * @param border_size The number of characters have already been written at the left border
	 * @param text The text to be aligned right
	 * @param color The color of the emited text
	*/
	inline void emitRightString(size_t border_size, const string& text, Colour color) const {
		const size_t emit_length = CATCH_CONFIG_CONSOLE_WIDTH - border_size - text.size();
		for (size_t i = 0ull; i < emit_length; i++) {
			this->stream << ' ';
		}
		this->stream << color << text << endl;
	}

	/**
	 * @brief Emit a bar of horizontal symbols with the width of the console
	*/
	inline void emitSymbol(const char symbol) const {
		for (int i = 0; i < CATCH_CONFIG_CONSOLE_WIDTH; i++) {
			this->stream << symbol;
		}

		this->stream << endl;
	}

	/**
	 * @brief Emit a summary line contains about the stats of an assertion
	 * @param assertion The assertion to be emitted
	*/
	inline void emitStats(const Counts& assertion) const {
		this->stream << Colour(Colour::Cyan) << assertion.total() << " Total |"
			<< Colour(Colour::ResultSuccess) << "| " << assertion.passed << " Passed |"
			<< Colour(Colour::ResultExpectedFailure) << "| " << assertion.failedButOk << " Warned |"
			<< Colour(Colour::ResultError) << "| " << assertion.failed << " Failed"
			<< endl;
	}

	/**
	 * @brief Emit a section, recursively including all its children sections
	 * @param section The root section to be emited
	 * @param depth The current recursion depth, start from 0
	*/
	void writeSection(const CumulativeReporterBase::SectionNode* section, unsigned short depth = 0u) const {
		const auto& sec_stats = section->stats;
		const string& sec_name = sec_stats.sectionInfo.name;

		//print the current section
		this->stream << sec_name;
		//print section pass status
		const bool allPassed = sec_stats.assertions.allPassed();
		if (allPassed) {
			this->emitRightString(sec_name.size(), "PASS", Colour(Colour::Success));
		}
		else if (sec_stats.assertions.allOk()) {
			this->emitRightString(sec_name.size(), "WARN", Colour(Colour::Warning));
		}
		else {
			this->emitRightString(sec_name.size(), "FAIL", Colour(Colour::Error));
		}

		//print message (if any)
		for (const auto& assertion : section->assertions) {
			//throw non-serious info
			if (!assertion.infoMessages.empty()) {
				this->stream << Colour(Colour::Red) << "Message stack trace:" << endl;
				for (const auto& info : assertion.infoMessages) {
					//the output color
					Colour::Code output_color;
					//choose output color
					switch (info.type) {
					case ResultWas::Info:
						//print info
						output_color = Colour::SecondaryText;
						break;
					case ResultWas::Warning:
					case ResultWas::Ok:
						//print warning
						output_color = Colour::Warning;
						break;
					default:
						//print error
						output_color = Colour::Error;
						break;
					}
					this->stream << "==>" << Colour(output_color) << info.message << endl;
					this->stream << "::" << info.lineInfo.line << '(' << info.lineInfo.file << ')' << endl;
				}
			}
			//bring up user's attention with error message
			if (!allPassed) {
				const auto& result = assertion.assertionResult;

				this->stream << Colour(Colour::Red) << "Caused by:" << endl;
				this->stream << result.getSourceInfo() << endl;
				this->stream << "Was expecting: " << Colour(Colour::Yellow) << result.getExpressionInMacro() << endl;
				this->stream << "Evaludated to: " << Colour(Colour::Yellow) << result.getExpandedExpression() << endl;
			}
		}
		
		this->stream << endl;
		//write all children
		for (const auto& child : section->childSections) {
			this->writeSection(child.get(), depth + 1u);
		}
	}

public:

	/**
	 * @brief Init STPConsoleReporter
	 * @param config Test report configuration
	*/
	STPConsoleReporter(const ReporterConfig& config) : CumulativeReporterBase(config) {

	}

	~STPConsoleReporter() = default;

	static string getDescription() {
		return "The default reporter for project super terrain +, with a nice layout of all test cases and sections";
	}

	/* Override some functions in CumulativeReporterBase */

	void noMatchingTestCases(const string& spec) override {
		this->stream << spec << endl;
	}

	//emit all results after the test run
	void testRunEndedCumulative() override {
		for (const auto& run : m_testRuns) {
			//write the information about the current run
			this->emitCentreString(run.value.runInfo.name);
			this->emitSymbol('=');

			//for each group in a run
			const auto& total_group = run.children;
			for (const auto& group : total_group) {
				//write information about the current group
				stringstream group_ss;
				const auto& group_info = group->value.groupInfo;
				group_ss << Colour(Colour::Headers) << "Group " << group_info.groupIndex << '/' << group_info.groupsCounts << " -- " << group_info.name;
				this->emitCentreString(group_ss.str());
				this->emitSymbol('.');
				
				//for each test case in a group
				const auto& total_case = group->children;
				for (const auto& testcase : total_case) {
					//write information about the current test case
					const auto* testcase_info = testcase->value.testInfo;
					this->emitSymbol('-');
					this->stream << Colour(Colour::FileName) << testcase_info->name << endl;
					this->emitSymbol('-');

					//for each section in a test case
					const auto& total_section = testcase->children;
					for (const auto& section : total_section) {
						//write a section recursively
						this->writeSection(section.get());
					}
					//end of section
					//emit stats for this test case
					this->emitStats(testcase->value.totals.assertions);
				}
				//end of test case
			}
			//end of group
		}
		//end of run
	}

};

CATCH_REGISTER_REPORTER("stp_console", STPConsoleReporter);