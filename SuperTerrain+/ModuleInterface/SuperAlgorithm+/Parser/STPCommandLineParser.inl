//TEMPLATE DEFINITION FOR COMMAND LINE PARSER
#ifdef _STP_COMMAND_LINE_PARSER_H_

//Exception
#include <SuperTerrain+/Exception/STPValidationFailed.h>

#include <memory>
#include <algorithm>

#define NAMESPACE_CMD_NAME SuperTerrainPlus::STPAlgorithm::STPCommandLineParser

/* ---------------------------- count ---------------------------- */
constexpr void NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement::set(const size_t num) noexcept {
	this->Min = num;
	this->Max = num;
}

/* ---------------------------- tree ----------------------------- */
#define TREE_BRANCH_TEMPLATE template<class T, size_t N>
#define TREE_BRANCH_NAME NAMESPACE_CMD_NAME::STPInternal::STPTreeBranch<T, N>

TREE_BRANCH_TEMPLATE
inline const T* TREE_BRANCH_NAME::begin() const noexcept {
	return std::addressof(*this->Leaf.cbegin());
}

TREE_BRANCH_TEMPLATE
inline const T* TREE_BRANCH_NAME::end() const noexcept {
	return std::addressof(*this->Leaf.cend());
}

TREE_BRANCH_TEMPLATE
inline size_t TREE_BRANCH_NAME::size() const noexcept {
	return this->Leaf.size();
}

#undef TREE_BRANCH_NAME
#undef TREE_BRANCH_TEMPLATE
/* --------------------------- command ------------------------------ */
inline bool NAMESPACE_CMD_NAME::STPInternal::STPBaseCommand::isGroup() const noexcept {
	return this->IsGroup;
}

inline bool NAMESPACE_CMD_NAME::STPInternal::STPBaseCommand::isSubcommand() const noexcept {
	return !this->IsGroup;
}

#define COMMAND_TEMPLATE template<size_t ON, size_t CN>
#define COMMAND_NAME NAMESPACE_CMD_NAME::STPCommand<ON, CN>

COMMAND_TEMPLATE
inline const NAMESPACE_CMD_NAME::STPInternal::STPBaseOptionTreeBranch& COMMAND_NAME::option() const noexcept {
	return this->Option;
}

COMMAND_TEMPLATE
inline const NAMESPACE_CMD_NAME::STPInternal::STPBaseCommandTreeBranch& COMMAND_NAME::command() const noexcept {
	return this->Command;
}

#undef COMMAND_NAME
#undef COMMAND_TEMPLATE
/* ------------------------------------------ option ------------------------------------------------------------ */
inline bool NAMESPACE_CMD_NAME::STPInternal::STPBaseOption::isPositional() const noexcept {
	return this->PositionalPrecedence > 0u;
}

#define OPTION_CONVERT(TEMP) inline void NAMESPACE_CMD_NAME::STPOption<TEMP>::convert([[maybe_unused]] const STPReceivedArgument& rx_arg) const

template<class BT>
OPTION_CONVERT(BT) {
	STP_ASSERTION_VALIDATION(rx_arg.size() == 1u, "The number of argument in a single-argument option must be exactly one");

	//just use our string tool to convert the argument
	*this->Variable = rx_arg.front().to<BT>();
}

OPTION_CONVERT(void) {
	//do nothing
}

OPTION_CONVERT(bool) {
	STP_ASSERTION_VALIDATION(rx_arg.size() <= 1u, "The number of argument in a flag option must be no more than one");
	
	//if there is an argument, read directly from it
	//a flag does not have any argument, simply assign from its inferred value
	*this->Variable = rx_arg.size() == 1u ? rx_arg.front().to<bool>() : this->InferredValue;
}

template<class VT>
OPTION_CONVERT(std::vector<VT>) {
	this->Variable->clear();
	this->Variable->resize(rx_arg.size());
	//convert each argument
	//we assume each type is a basic convertible type, not something like a vector of vector
	//so we can avoid recursive check
	std::transform(rx_arg.cbegin(), rx_arg.cend(), this->Variable->cbegin(), [](const auto& arg) { return arg.to<VT>(); });
}

template<class... TT>
OPTION_CONVERT(std::tuple<TT...>) {
	STP_ASSERTION_VALIDATION(rx_arg.size() == std::tuple_size_v<std::tuple<TT...>>,
		"The number of argument must equal to the number of space in a space");

	//TODO: capture `auto` as template argument in C++ 20 so it's less verbose
	auto convertOne = [i = static_cast<size_t>(0u), &rx_arg](auto& dst) mutable -> void {
		dst = rx_arg[i++].to<std::remove_reference_t(decltype(dst))>();
	};
	//iterate over each element in the tuple
	std::apply([&convertOne](auto&... arg) { (convertOne(arg), ...); }, *this->Variable);
}

#undef OPTION_CONVERT
#undef CHECK_BINDING
/* --------------------------------------------------------------------------------------------------------------- */

#undef NAMESPACE_CMD_NAME

#endif//_STP_COMMAND_LINE_PARSER_H_