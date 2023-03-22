#pragma once
#ifndef _STP_COMMAND_LINE_PARSER_H_
#define _STP_COMMAND_LINE_PARSER_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
//Parser Framework
#include "./Framework/STPBasicStringAdaptor.h"

//Container
#include <array>
#include <vector>
#include <tuple>
#include <utility>
//String
#include <string>
#include <string_view>
#include <ostream>

#include <cstddef>
#include <type_traits>
#include <exception>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPCommandLineParser provides a flexible interface for programmers to obtain user-configuration from command line.
	*/
	namespace STPCommandLineParser {

		//internal utility used by the command line parser
		namespace STPInternal {

			//The range of a number of counts as an requirement.
			struct STPCountRequirement {
			public:

				//The minimum and maximum number of count in the requirement.
				size_t Min, Max;

				/**
				 * @brief Default initialise the count value to all zero.
				*/
				constexpr STPCountRequirement() noexcept;

				~STPCountRequirement() = default;

				/**
				 * @brief Set the range to a fixed number.
				 * @param num The number to be set, such that min = max = num.
				*/
				constexpr void set(size_t) noexcept;

				/**
				 * @brief Set the maximum number of count to be unlimited.
				*/
				constexpr void unlimitedMax() noexcept;

				/**
				 * @brief Check if the max value is unlimited.
				 * @return True if max is unlimited.
				*/
				constexpr bool isMaxUnlimited() const noexcept;

			};

			class STPBaseOption;
			class STPBaseCommand;

			/**
			 * @brief STPBaseTreeBranch is a visitor to different type of tree branch with different size.
			 * @tparam T The type of the leaf.
			*/
			template<class T>
			class STPBaseTreeBranch {
			public:

				STPBaseTreeBranch() = default;

				virtual ~STPBaseTreeBranch() = default;

				//the begin iterator
				virtual const T* begin() const noexcept = 0;

				//the end iterator
				virtual const T* end() const noexcept = 0;

				//the number of node of a branch
				virtual size_t size() const noexcept = 0;

				//check if the container is empty
				virtual bool empty() const noexcept = 0;

			};
			typedef STPBaseTreeBranch<const STPBaseOption*> STPBaseOptionTreeBranch;
			typedef STPBaseTreeBranch<const STPBaseCommand*> STPBaseCommandTreeBranch;

			/**
			 * @brief STPTreeBranch is an implementation of tree branch with a specific type and size.
			 * @tparam T The type of the branch element.
			 * @param N The number of element a branch can hold.
			 * @see STPBaseTreeBranch
			*/
			template<class T, size_t N>
			struct STPTreeBranch : public STPBaseTreeBranch<T> {
			public:

				std::array<T, N> Leaf;

				STPTreeBranch() = default;

				/**
				 * @brief Create the tree branch by directly initialising from an array of child nodes.
				 * @param leaf A array of child nodes.
				*/
				constexpr STPTreeBranch(std::array<T, N>) noexcept;

				~STPTreeBranch() override = default;

				const T* begin() const noexcept override;
				const T* end() const noexcept override;
				size_t size() const noexcept override;
				bool empty() const noexcept override;

			};
			template<size_t N>
			using STPOptionTreeBranch = STPTreeBranch<const STPBaseOption*, N>;
			template<size_t N>
			using STPCommandTreeBranch = STPTreeBranch<const STPBaseCommand*, N>;

			/**
			 * @brief STPBaseOption defines an option and its common properties in the command line.
			*/
			class STPBaseOption {
			public:

				//Recognised arguments for the current option.
				typedef std::vector<STPStringViewAdaptor> STPReceivedArgument;
				//TODO: use span in C++ 20 on the received array of arguments
				//Just to emulate a span for the vector type.
				typedef std::pair<STPReceivedArgument::const_iterator, STPReceivedArgument::const_iterator> STPReceivedArgumentSpan;

			protected:

				/**
				 * @brief Create an exception when a conversion error happens.
				 * @param custom Some custom error message to be added.
				 * @param rx_arg The pointer to the array of arguments that causes the problem.
				*/
				[[noreturn]] STP_ALGORITHM_HOST_API static void throwConversionError(const char*, const STPReceivedArgument&);

			public:

				//Specify the name of the option.
				//A short name begins with a single control symbol, and can only be followed by one character.
				//A long name begins with double control symbols, and can be a combination of any alphabet and the control symbol itself.
				//In the name here, do not include the control symbol prefix.
				std::string_view ShortName, LongName;
				//An optional message to appear in the command line when printing the help message for the option.
				std::string_view Description;

				//Specifies the minimum and maximum number of argument expected from this option.
				//For a positional option, due to its uncertain nature, it is recommended to have a fixed argument count
				//	for an option that is not the last positional option in a subcommand.
				STPCountRequirement ArgumentCount;

				//A positional argument allows omitting option name when specified in the command line,
				//however it is order-sensitive, and should not follow any option that takes a variable number of argument.
				//A precedence specifies their order, and lower number gets higher precedence.
				//If the precedence is zero, this option is treated as a standard, non-positional option.
				//If any two positional options in the same subcommand come with the same, non-zero precedence, the order is unspecified.
				unsigned int PositionalPrecedence = 0u;
				//Specifies, if any not-the-last positional options should take positional arguments in a greedy way,
				//such that it will take as many arguments as it can as defined in the variable how many it can take at max.
				//Otherwise positional arguments are taken in a lazy manner, i.e., as few as possible.
				//This option has no effect for non-positional argument.
				bool PositionalGreedy = true;
				//Set to true to enforce that this option is expected from the command line.
				bool Require = false;

				//Specifies the delimiter if the arguments are provided using a delimiter style.
				//Define delimiter as the null character to disable delimiter style.
				char Delimiter = '\0';

				//Result set by the parser. They should be left as default value by the application before parsing.
				mutable struct {
				public:

					//True if the option has been parsed successfully from the command line.
					bool IsUsed = false;

				} Result;

				STPBaseOption() = default;

				virtual ~STPBaseOption() = default;

				/**
				 * @brief Convert the argument in string form into user-expected format.
				 * @param rx_arg The arguments from the command line for the current option.
				*/
				virtual void convert(const STPReceivedArgument&) const = 0;

				/**
				 * @brief Check if the current option is positional.
				 * @return True if the option takes positional arguments.
				*/
				bool isPositional() const noexcept;

				/**
				 * @brief Check if the current option supports specifying arguments using delimiter.
				 * @return True if it supports.
				*/
				bool supportDelimiter() const noexcept;

				/**
				 * @brief Check if the current option is used in the command line, i.e., user has specified this option.
				 * @return True if the option is used.
				*/
				bool used() const noexcept;

			};

			/**
			 * @brief STPBaseCommand is a visitor to a user command definition.
			*/
			class STPBaseCommand {
			public:

				//The identifier name for this command.
				//For a group, the name is simply for debug purposes.
				std::string_view Name;
				//a help message to tell user how to use this command.
				std::string_view Description;
				//The minimum and maximum number of option, which is the sum of all parsed options and parsed group.
				//Each parsed group is counted as 1 regardless of how many options have been recognised within.
				STPCountRequirement OptionCount;

				//Specify if this command is a group rather than a subcommand.
				//A group inherits everything from its parent command.
				//A group cannot contain any subcommand, which will be simply ignored without notice.
				//Otherwise, this is considered as a subcommand that is totally independent with its parent.
				//A subcommand needs to be specified by its name in the command line.
				bool IsGroup = false;

				STPBaseCommand() = default;

				virtual ~STPBaseCommand() = default;

				//get all options in the current command
				virtual const STPBaseOptionTreeBranch& option() const noexcept = 0;

				//get all children group/subcommand
				virtual const STPBaseCommandTreeBranch& command() const noexcept = 0;

				/**
				 * @brief Test if the current command is a group.
				 * @return True if it is a group.
				*/
				bool isGroup() const noexcept;

				/**
				 * @brief Test if the current command is a subcommand of a parent command.
				 * @return True if it is a subcommand.
				*/
				bool isSubcommand() const noexcept;

			};

			/**
			 * @brief STPArgumentConverter is a utility to convert a number of string arguments to binding variable.
			 * The base version only supports conversion from fundamental types.
			 * This utility can be specialised to support more complex types.
			 * @tparam T The type of the binding variable.
			*/
			template<typename T>
			struct STPArgumentConverter {
			public:

				/**
				 * @brief Invoke the argument converter.
				 * @param rx_arg The pointer to an array of received arguments.
				 * @param var The pointer to the binding variable output.
				 * @return The number of argument converted.
				 * @exception Any exception generated from the fundamental type conversion from `STPBasicStringAdaptor`.
				 * This will happen if the argument has any syntactic error.
				 * In other non-fatal, recoverable cases, use return value of 0 to indicate nothing converted.
				*/
				size_t operator()(const STPBaseOption::STPReceivedArgumentSpan&, T&) const;

			};
			//A specialisation for void, which defines no converter because there is nothing to be converted.
			template<>
			struct STPArgumentConverter<void> { };

		}

#define OPTION_CLASS_DEF(TYPE) template<class Conv> struct STPOption<TYPE, Conv> : public STPInternal::STPBaseOption

#define OPTION_CLASS_BASE_MEMBER \
STPOption() = default; \
~STPOption() override = default; \
using STPBaseOption::STPReceivedArgument; \
void convert(const STPReceivedArgument&) const override

#define OPTION_CLASS_MEMBER(VAR_TYPE) \
private: \
const Conv Converter; \
public: \
VAR_TYPE* Variable = nullptr; \
STPOption(VAR_TYPE&, Conv&& = Conv {}); \
OPTION_CLASS_BASE_MEMBER
		
		/**
		 * @brief STPOption allows binding variable to an option and parsed from the command line input.
		 * The constructor can take a reference to the binding variable, and optionally the custom defined converter;
		 * this can be used to deduce the class template.
		 * @tparam BT The binding variable type. Can be any fundamental type, a vector type, or a tuple.
		 * If the type if void, no binding variable is required.
		 * If a binding variable is used, the pointer to the binding variable should be not null,
		 * and its lifetime should be preserved until the end of the option's lifetime.
		 * @tparam Conv The argument converter to be used, or using the default one if not provided.
		 * @see STPBaseOption
		*/
		template<class BT, class Conv = STPInternal::STPArgumentConverter<BT>>
		struct STPOption : public STPInternal::STPBaseOption {

			OPTION_CLASS_MEMBER(BT);

		};
		//No binding variable.
		OPTION_CLASS_DEF(void) {

			OPTION_CLASS_BASE_MEMBER;

		};
		//A specialisation for a flag.
		OPTION_CLASS_DEF(bool) {

			//If the flag appears on the command line without argument, this is the value set to the binding variable.
			//Otherwise the variable is set to the argument parsed from the command line.
			bool InferredValue = true;

			OPTION_CLASS_MEMBER(bool);

		};

#undef OPTION_CLASS_DEF
#undef OPTION_CLASS_MEMBER
#undef OPTION_CLASS_BASE_MEMBER

		/**
		 * @brief STPCommand is a collection of options, groups and subcommands.
		*/
		template<size_t ON, size_t CN>
		struct STPCommand : public STPInternal::STPBaseCommand {
		private:

			//convert tuple of reference to derived tree branch to pointers of base tree branch
			template<class Base, typename TupLeaf>
			auto toTreeBranch(const TupLeaf&) noexcept;

		public:

			//Defines all options.
			STPInternal::STPOptionTreeBranch<ON> Option;
			//Defines all subcommands and groups.
			STPInternal::STPCommandTreeBranch<CN> Command;

			STPCommand() = default;

			/**
			 * @brief Create a command with its option and command members.
			 * This can automatically deduce the size of the tree branches.
			 * @tparam Opt... The type of each derived option.
			 * @tparam Cmd... The type of each derived command.
			 * @param tup_option Tuple of reference to options.
			 * @param tup_command Tuple of reference to command, can be group or subcommand.
			*/
			template<class... Opt, class... Cmd>
			STPCommand(std::tuple<Opt&...>, std::tuple<Cmd&...>) noexcept;

			~STPCommand() override = default;

			const STPInternal::STPBaseOptionTreeBranch& option() const noexcept override;
			const STPInternal::STPBaseCommandTreeBranch& command() const noexcept override;

		};
		template<class... Opt, class... Cmd>
		STPCommand(std::tuple<Opt&...>, std::tuple<Cmd&...>) -> STPCommand<sizeof...(Opt), sizeof...(Cmd)>;

		/**
		 * @brief STPParseResult contains return value from the command line parser.
		*/
		struct STPParseResult {
		public:

			/**
			 * @brief STPHelpPrinterData contains information from the parser to print the help message.
			*/
			struct STPHelpPrinterData {
				//The name of the program, which appears as the first value in the command line option.
				std::string_view ProgramName;
				//The sequence of subcommands recognised from the command line.
				std::vector<const STPInternal::STPBaseCommand*> CommandPath;
			} HelpData;

			//Parsing results are all stored to each option, but the requirements may not be satisfied.
			//This contains the status of the post-parsing validation.
			//Pointer is null is validation is successful, otherwise it contains the exception generated.
			//Separating validation exception allows application to make certain adjustment,
			//for example ignore post-validation if some overriding options are specified, e.g., help option.
			std::exception_ptr ValidationStatus;
			//A boolean value specifies if the parser has parsed any option.
			//False if there is no command line option specified in the input.
			bool NonEmptyCommandLine;

			/**
			 * @brief Get the pointer to command that appears as the last subcommand in the command line.
			 * This was the subcommand branch the parser worked on.
			 * @return The pointer to the command branch.
			*/
			const STPInternal::STPBaseCommand& commandBranch() const noexcept;
			
		};

		/**
		 * @brief STPHelpPrinter formats command line parser rules to a readable format.
		 * This printer can be fed directly to a stream output.
		*/
		struct STPHelpPrinter {
		public:

			//The return value from the parser.
			const STPParseResult::STPHelpPrinterData* PrinterData;

			//The number of space put before printing each option.
			size_t IndentationWidth;
			//Specifies the number of character for a summary line.
			size_t SummaryLineWidth;
			//Specifies the number of character a detail usage message line can hold.
			size_t DetailLineWidth;

		};

		/**
		 * @brief Validate if a command line parsing rule set is valid.
		 * This feature is intended to be used by developer during testing,
		 * and should be removed in release code due to high cost of validation.
		 * Throw exception if validation fails with description of the error.
		 * @param command The root node to the command line parser configuration.
		*/
		STP_ALGORITHM_HOST_API void validate(const STPInternal::STPBaseCommand&);

		/**
		 * @brief Print formatted help message of a command line parsing rule set to a stream, with a help message printer.
		*/
		STP_ALGORITHM_HOST_API std::ostream& operator<<(std::ostream&, const STPHelpPrinter&);

		/**
		 * @brief Pre-process the command line arguments by encoding them in an implementation-defined manner.
		 * The encoded string can then be used for parsing.
		 * @param argc The number of argument.
		 * @param argv The value of arguments.
		 * @return The encoded string.
		 * It is application's responsibility to maintain the lifetime of this string.
		*/
		STP_ALGORITHM_HOST_API std::string encode(int, const char* const*);

		/**
		 * @brief Start the command line parser.
		 * It is recommended that the parsing rule is validated during development, the parser will not check for any error,
		 * and parsing result is non-deterministic if the rule is invalid.
		 * @param arg_encoded The output from the command line encoder.
		 * @param app_name The name of the application using this parser, this is for debug purposes only.
		 * @param command Specify the root node to the command line parser configuration.
		 * All parsing result will be returned back to here at the end of parsing.
		 * @return The parsing result.
		*/
		STP_ALGORITHM_HOST_API STPParseResult parse(const std::string&, const std::string_view&, const STPInternal::STPBaseCommand&);

	}

}
#include "STPCommandLineParser.inl"
#endif//_STP_COMMAND_LINE_PARSER_H_