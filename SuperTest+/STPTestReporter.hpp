#pragma once
#ifndef _STP_TEST_REPORTER_HPP_
#define _STP_TEST_REPORTER_HPP_

//Base reporter
#include "catch2/catch.hpp"

namespace Catch {

	struct STPTestReporter : public CumulativeReporterBase<STPTestReporter> {
	private:

		void print_section(const SectionNode& sec) {
			const auto& sec_stats = sec.stats;
			const std::string last_name = trim(sec_stats.sectionInfo.name);

			stream << last_name;
			for (int i = 0; i < CATCH_CONFIG_CONSOLE_WIDTH - last_name.length() - 4; i++) {
				stream << " ";
			}

			if (sec_stats.assertions.allPassed()) {
				stream << Colour(Colour::Success) << "PASS\n\n";

				//print warning messages
				for (auto ass = sec.assertions.begin(); ass != sec.assertions.end(); ass++) {
					if (!ass->infoMessages.empty()) {
						stream << Colour(Colour::Red) << "Message stack trace:\n";
						for (auto info = ass->infoMessages.begin(); info != ass->infoMessages.end(); info++) {
							stream << "At: " << info->lineInfo.line << ", in: " << info->lineInfo.file << "\n";
							stream << "==>" << Colour(Colour::Yellow) << info->message << "\n";
						}
					}
				}
			}
			else {
				if (sec.stats.assertions.allOk()) {
				stream << Colour(Colour::Warning) << "WARN\n";
				}
				else {
					stream << Colour(Colour::Error) << "FAIL\n";
				}
				
				//print assertion message
				for (auto ass = sec.assertions.begin(); ass != sec.assertions.end(); ass++) {
					if (!ass->infoMessages.empty()) {
						stream << Colour(Colour::Red) << "Message stack trace:\n";
						for (auto info = ass->infoMessages.begin(); info != ass->infoMessages.end(); info++) {
							stream << "At: " << info->lineInfo.line << ", in: " << info->lineInfo.file << "\n";
							switch (info->type) {
							case ResultWas::Info: stream << "==>" << Colour(Colour::LightGrey) << info->message << "\n";
								break;
							case ResultWas::Warning: stream << "==>" << Colour(Colour::Warning) << info->message << "\n";
								break;
							default: stream << "==>" << Colour(Colour::Error) << info->message << "\n";
								break;
							}
						}
					}
					stream << Colour(Colour::Red) << "Caused by:\n";
					stream << ass->assertionResult.getSourceInfo() << "\n";
					stream << "Was expecting: " << Colour(Colour::Yellow) << ass->assertionResult.getExpressionInMacro() << "\n";
					stream << "Evaludated to: " << Colour(Colour::Yellow) << ass->assertionResult.getExpandedExpression() << "\n";
				}

				stream << "\n";
			}
 
			for (const auto& child : sec.childSections) {
				print_section(*child);
			}
		}

	public:

		STPTestReporter(const ReporterConfig& config) : CumulativeReporterBase<STPTestReporter>(config){

		}

		~STPTestReporter() = default;

		static std::string getDescription() {
			return "The default reporter for project super terrain +, with a nice layout of all test cases and sections";
		}

		void noMatchingTestCases(const std::string&) override {

		}

		void testRunEndedCumulative() override {
			
		}

		void testGroupEnded(const TestGroupStats& group) override {
			for (auto cases = this->m_testCases.begin(); cases != this->m_testCases.end(); cases++) {
				//writing test cases starting
				const auto& case_stats = cases->get()->value;
				stream << Colour(Colour::FileName) << case_stats.testInfo.name << "\n";
				for (int i = 0; i < CATCH_CONFIG_CONSOLE_WIDTH; i++) {
					stream << "=";
				}
				stream << "\n";

				//writing section
				const auto& sections = cases->get()->children.front();
				print_section(*sections);

				//writing test cases ending
				stream << Colour(Colour::Cyan) << case_stats.totals.assertions.total() << " Total |"
					<< Colour(Colour::ResultSuccess) << "| " << case_stats.totals.assertions.passed << " Passed |" 
					<< Colour(Colour::Warning) << "| " << case_stats.totals.assertions.failedButOk << " Warned |"
					<< Colour(Colour::ResultError) << "| " << case_stats.totals.assertions.failed << " Failed\n";

				for (int i = 0; i < CATCH_CONFIG_CONSOLE_WIDTH; i++) {
					stream << "-";
				}
				stream << "\n";

			}
		}
	};

	CATCH_REGISTER_REPORTER("stp", STPTestReporter);

}

#endif//_STP_TEST_REPORTER_HPP_