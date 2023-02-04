#pragma once
#ifndef _STP_COMMAND_LINE_PARSER_H_
#define _STP_COMMAND_LINE_PARSER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
//Parser Framework
#include "./Framework/STPBasicStringAdaptor.h"

//Container
#include <array>
#include <vector>
#include <tuple>
//String
#include <string>
#include <string_view>
#include <ostream>

#include <cstddef>
#include <type_traits>

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
				 * @brief Set the range to a fixed number.
				 * @param num The number to be set, such that min = max = num.
				*/
				constexpr void set(size_t) noexcept;

			};

			class STPBaseOption;
			class STPBaseCommand;

			/**
			 * @brief STPBaseTreeBranch is a visitor to different type of tree branch with different size.
			 * @tparam T The type of the tree branch.
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
				 * @brief Initialise the tree branch with reference to type, and store their pointers.
				 * @tparam Node... The derived type of each node.
				 * @param n... Reference of each node.
				*/
				template<class... Node, typename =
					std::enable_if_t<
						std::conjunction_v<
							std::is_base_of<
								std::remove_pointer_t<T>,
								Node
							>...
						>
					>
				>
				STPTreeBranch(const Node&... n) noexcept : Leaf { &n... } {

				}

				~STPTreeBranch() override = default;

				const T* begin() const noexcept override;
				const T* end() const noexcept override;
				size_t size() const noexcept override;

			};

			/**
			 * @brief STPBaseOption defines an option and its common properties in the command line.
			*/
			class STPBaseOption {
			public:

				//Recognised arguments for the current option.
				typedef std::vector<STPStringViewAdaptor> STPReceivedArgument;

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
				unsigned int PositionalPrecedence;
				//Specifies, if any not-the-last positional options should take positional arguments in a greedy way,
				//such that it will take as many arguments as it can as defined in the variable how many it can take at max.
				//Otherwise positional arguments are taken in a lazy manner, i.e., as few as possible.
				//This option has no effect for non-positional argument.
				bool PositionalGreedy;
				//Set to true to enforce that this option is expected from the command line.
				bool Require;

				//Specifies the delimiter if the arguments are provided using a delimiter style.
				char Delimiter;

				//Result set by the parser. They should be left as default value by the application before parsing.
				mutable struct {
				public:

					//True if the option has been parsed successfully from the command line.
					bool IsUsed;

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

				//Specify if this subcommand or group is required.
				//Ignored if the command if the root.
				bool Require;
				//Specify if this command is a group rather than a subcommand.
				//A group inherits everything from its parent command.
				//A group cannot contain any subcommand, which will be simply ignored without notice.
				//Otherwise, this is considered as a subcommand that is totally independent with its parent.
				//A subcommand needs to be specified by its name in the command line.
				bool IsGroup;

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

		}

		template<size_t N>
		using STPOptionTreeBranch = STPInternal::STPTreeBranch<const STPInternal::STPBaseOption*, N>;
		template<size_t N>
		using STPCommandTreeBranch = STPInternal::STPTreeBranch<const STPInternal::STPBaseCommand*, N>;

#define OPTION_CLASS_BASE_MEMBER \
STPOption() = default; \
~STPOption() override = default; \
using STPBaseOption::STPReceivedArgument; \
void convert(const STPReceivedArgument&) const override

#define OPTION_CLASS_MEMBER(VAR_TYPE) \
mutable VAR_TYPE Variable; \
OPTION_CLASS_BASE_MEMBER
		
		/**
		 * @brief STPOption allows binding variable to an option and parsed from the command line input.
		 * @tparam BT The binding variable type. Can be any fundamental type, a vector type, or a tuple.
		 * If the type if void, no binding variable is required.
		 * If a binding variable is used, the pointer to the binding variable should be not null,
		 * and its lifetime should be preserved until the end of the option's lifetime.
		 * @see STPBaseOption
		*/
		template<class BT>
		struct STPOption : public STPInternal::STPBaseOption {
		public:

			OPTION_CLASS_MEMBER(BT*);

		};
		//No binding variable.
		template<>
		struct STPOption<void> : public STPInternal::STPBaseOption {
		public:

			OPTION_CLASS_BASE_MEMBER;

		};
		//A specialisation for a flag.
		template<>
		struct STPOption<bool> : public STPInternal::STPBaseOption {
		public:

			//If the flag appears on the command line without argument, this is the value set to the binding variable.
			//Otherwise the variable is set to the argument parsed from the command line.
			bool InferredValue;

			OPTION_CLASS_MEMBER(bool*);

		};
		//A specialisation of the option whose binding variable is a vector.
		template<class VT>
		struct STPOption<std::vector<VT>> : public STPInternal::STPBaseOption {
		public:

			OPTION_CLASS_MEMBER(std::vector<VT>*);

		};
		//A specialisation of the option whose binding variable is a tuple.
		template<class... TT>
		struct STPOption<std::tuple<TT...>> : public STPInternal::STPBaseOption {
		public:

			OPTION_CLASS_MEMBER(std::tuple<TT...>*);

		};

#undef OPTION_CLASS_MEMBER
#undef OPTION_CLASS_BASE_MEMBER

		/**
		 * @brief STPCommand is a collection of options, groups and subcommands.
		*/
		template<size_t ON, size_t CN>
		struct STPCommand : public STPInternal::STPBaseCommand {
		public:

			//Defines all options.
			STPOptionTreeBranch<ON> Option;
			//Defines all subcommands and groups.
			STPCommandTreeBranch<CN> Command;

			STPCommand() = default;

			~STPCommand() override = default;

			const STPInternal::STPBaseOptionTreeBranch& option() const noexcept override;
			const STPInternal::STPBaseCommandTreeBranch& command() const noexcept override;

		};

		/**
		 * @brief STPParseResult contains return value from the command line parser.
		*/
		struct STPParseResult {
		public:

			//The name of the program, which appears as the first value in the command line option.
			std::string_view ProgramName;
			//The sequence of subcommands recognised from the command line.
			std::vector<const STPInternal::STPBaseCommand*> CommandPath;
			//A boolean value specifies if the parser has parsed any option.
			//False if there is no command line option specified in the input.
			bool NonEmptyCommandLine;
			

		};

		/**
		 * @brief STPHelpPrinter formats command line parser rules to a readable format.
		 * This printer can be fed directly to a stream output.
		*/
		struct STPHelpPrinter {
		public:

			//The return value from the parser.
			const STPParseResult* ParseResult;

			//The number of space put before printing each option.
			size_t IndentationWidth;
			//Specifies the maximum number of character a line can hold.
			size_t MaxLineWidth;

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