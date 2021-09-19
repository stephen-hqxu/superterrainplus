#pragma once

//System
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>
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
using std::string_view;

/**
 * @brief STPConsoleReporter is a test reporter for SuperTerrain+, it outputs a human-readable test report and is best suited for console output.
*/
class STPConsoleReporter final : public CumulativeReporterBase {
private:

	template<class S>
	constexpr static inline string_view getView(const S& str) {
		static_assert(std::disjunction_v<std::is_same<S, string>, std::is_same<S, Catch::StringRef>>);
		return string_view(str.data());
	}

	/**
	 * @brief Emit text that exceeds the width of the console and auto wrap it to the next line.
	 * @param text The text to wrap.
	 * @param reserve Control the number of maximum character offset that can be in one line.
	 * 0 is the default, means no offset
	 * If text does not exceed the line limit, it will simply emit the text
	 * @return The last line of the wrap
	*/
	string_view emitWrapped(string_view text, size_t reserve = 0ull) const {
		while (text.size() > CATCH_CONFIG_CONSOLE_WIDTH - reserve) {
			//we do not wish to break a single word into half, instead of wrap it when it's a space
			//find the last space in the emit string
			const size_t emitStart = text.rfind(' ', CATCH_CONFIG_CONSOLE_WIDTH - reserve);
			if (emitStart == string_view::npos) {
				//the entire width of text has no space, simply break the text
				const size_t breakLength = CATCH_CONFIG_CONSOLE_WIDTH - reserve - 1ull;
				const string emit = string(text.substr(0ull, breakLength)) + '-';
				//prune thet original string
				text.remove_prefix(breakLength);
				
				//output
				this->stream << emit << endl;
				continue;
			}
			const string_view emit = text.substr(0ull, emitStart + 1ull);
			//prune
			text.remove_prefix(emit.size());

			//output
			this->stream << emit << endl;
		}

		//return the text if wrap is not required
		return text;
	}

	/**
	 * @brief Emit a text to the stream that is centred at the console
	 * @param text The text to be centred
	*/
	void emitCentreString(const string_view& text) const {
		auto centreStr = [&stream = this->stream](const string_view& text) -> void {
			auto emit_border = [&stream](size_t border_size) -> void {
				for (size_t i = 0ull; i < border_size / 2ull; i++) {
					stream << ' ';
				}
			};
			//space left empty
			const size_t border = CATCH_CONFIG_CONSOLE_WIDTH - text.size();

			//emit the left border
			emit_border(border);

			//emit the text
			stream << text;

			//emit the right border
			emit_border(border);

			stream << endl;
		};

		if (text.size() > CATCH_CONFIG_CONSOLE_WIDTH) {
			//cannot centre the tetx because it's too wide, wrap it
			const string_view lastLine = this->emitWrapped(text);
			//and centre the last line
			centreStr(lastLine);
			return;
		}
		//simply centre the line
		centreStr(text);
	}

	/**
	 * @brief Emit a text that is aligned to the right border of the console
	 * @param border_size The number of characters have already been written at the left border
	 * @param text The text to be aligned right
	 * @param color The color of the emited text
	*/
	inline void emitRightString(size_t border_size, const string_view& text, Colour color) const {
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
		//reserve some spaces some the status tag at the end
		const string indented_sec = string(depth * 2u, '-') + "> " + sec_stats.sectionInfo.name;
		const string_view sec_name = this->emitWrapped(STPConsoleReporter::getView(indented_sec), 6ull);

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
					this->stream << info.lineInfo << endl;
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

	constexpr static inline char* getDescription() {
		return "The default reporter for project super terrain +, with a nice layout of all test cases and sections";
	}

	/* Override some functions in CumulativeReporterBase */

	//emit all results after the test run
	void testRunEndedCumulative() override {
		//begin a run
		this->emitSymbol('=');

		const auto* run = m_testRun.get();
		//write the information about the current run
		this->emitCentreString(STPConsoleReporter::getView(run->value.runInfo.name));
		this->emitSymbol('.');

		//for each test case in a run
		const auto& total_case = run->children;
		for (const auto& testcase : total_case) {
			//write information about the current test case
			const auto* testcase_info = testcase->value.testInfo;
			this->emitSymbol('-');
			stringstream testcase_ss;
			testcase_ss << Colour(Colour::FileName) << testcase_info->name;
			this->stream << this->emitWrapped(STPConsoleReporter::getView(testcase_ss.str())) << endl;
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
		//emit stats for the run
		this->emitSymbol('.');
		this->emitStats(run->value.totals.assertions);

		//end of run
		this->emitSymbol('=');
	}

};

CATCH_REGISTER_REPORTER("stp_console", STPConsoleReporter);