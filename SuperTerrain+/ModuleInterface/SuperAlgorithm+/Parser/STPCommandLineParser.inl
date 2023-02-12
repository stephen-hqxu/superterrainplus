//TEMPLATE DEFINITION FOR COMMAND LINE PARSER
#ifdef _STP_COMMAND_LINE_PARSER_H_

//Exception
#include <SuperTerrain+/Exception/STPValidationFailed.h>

#include <algorithm>
#include <limits>

#define NAMESPACE_CMD_NAME SuperTerrainPlus::STPAlgorithm::STPCommandLineParser

/* ---------------------------- count ---------------------------- */
constexpr NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement::STPCountRequirement() noexcept : Min(0u), Max(0u) {

}

constexpr void NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement::set(const size_t num) noexcept {
	this->Min = num;
	this->Max = num;
}

constexpr void NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement::unlimitedMax() noexcept {
	this->Max = std::numeric_limits<size_t>::max();
}

constexpr bool NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement::isMaxUnlimited() const noexcept {
	return this->Max == std::numeric_limits<size_t>::max();
}

/* ---------------------------- tree ----------------------------- */
#define TREE_BRANCH_TEMPLATE template<class T, size_t N>
#define TREE_BRANCH_NAME NAMESPACE_CMD_NAME::STPInternal::STPTreeBranch<T, N>

TREE_BRANCH_TEMPLATE
constexpr TREE_BRANCH_NAME::STPTreeBranch(const std::array<T, N> leaf) noexcept : Leaf(leaf) {

}

TREE_BRANCH_TEMPLATE
inline const T* TREE_BRANCH_NAME::begin() const noexcept {
	return this->Leaf.data();
}

TREE_BRANCH_TEMPLATE
inline const T* TREE_BRANCH_NAME::end() const noexcept {
	return this->Leaf.data() + this->Leaf.size();
}

TREE_BRANCH_TEMPLATE
inline size_t TREE_BRANCH_NAME::size() const noexcept {
	return this->Leaf.size();
}

TREE_BRANCH_TEMPLATE
inline bool TREE_BRANCH_NAME::empty() const noexcept {
	return this->Leaf.empty();
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
template<class... Opt, class... Cmd>
inline COMMAND_NAME::STPCommand(const std::tuple<Opt&...> tup_option, const std::tuple<Cmd&...> tup_command) noexcept :
	Option(STPCommand::toTreeBranch<STPInternal::STPBaseOption>(tup_option)),
	Command(STPCommand::toTreeBranch<STPInternal::STPBaseCommand>(tup_command)) {

}

COMMAND_TEMPLATE
template<class Base, typename TupLeaf>
inline auto COMMAND_NAME::toTreeBranch(const TupLeaf& tup_leaf) noexcept {
	return std::apply([](const auto&... leaf)
		//specify the array type parameter, so it works if tuple if empty
		{ return std::array<const Base*, std::tuple_size_v<TupLeaf>> { &static_cast<const Base&>(leaf)... }; }, tup_leaf);
}

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

inline bool NAMESPACE_CMD_NAME::STPInternal::STPBaseOption::supportDelimiter() const noexcept {
	return this->Delimiter != '\0';
}

inline bool NAMESPACE_CMD_NAME::STPInternal::STPBaseOption::used() const noexcept {
	return this->Result.IsUsed;
}

#define OPTION_CONVERT(TEMP) inline void NAMESPACE_CMD_NAME::STPOption<TEMP>::convert([[maybe_unused]] const STPReceivedArgument& rx_arg) const
#define OPTION_CTOR(TEMP) inline NAMESPACE_CMD_NAME::STPOption<TEMP>::STPOption(TEMP& v) : Variable(&v)

template<class BT>
OPTION_CTOR(BT) { }

template<class BT>
OPTION_CONVERT(BT) {
	const size_t arg_count = rx_arg.size();
	STP_ASSERTION_VALIDATION(arg_count <= 1u, "The number of argument in a single-argument option must be no more than one");

	if (arg_count == 1u) {
		//just use our string tool to convert the argument
		*this->Variable = rx_arg.front().to<BT>();
	}
}

OPTION_CONVERT(void) {
	//do nothing
}

OPTION_CTOR(bool) { }

OPTION_CONVERT(bool) {
	const size_t arg_count = rx_arg.size();
	STP_ASSERTION_VALIDATION(arg_count <= 1u, "The number of argument in a flag option must be no more than one");
	
	//if there is an argument, read directly from it
	//a flag does not have any argument, simply assign from its inferred value
	*this->Variable = arg_count == 1u ? rx_arg.front().to<bool>() : this->InferredValue;
}

template<class VT>
OPTION_CTOR(std::vector<VT>) { }

template<class VT>
OPTION_CONVERT(std::vector<VT>) {
	this->Variable->clear();
	this->Variable->resize(rx_arg.size());
	//convert each argument
	//we assume each type is a basic convertible type, not something like a vector of vector
	//so we can avoid recursive check
	std::transform(rx_arg.cbegin(), rx_arg.cend(), this->Variable->begin(), [](auto& arg) { return arg.template to<VT>(); });
}

template<class... TT>
OPTION_CTOR(std::tuple<TT...>) { }

template<class... TT>
OPTION_CONVERT(std::tuple<TT...>) {
	STP_ASSERTION_VALIDATION(rx_arg.size() == std::tuple_size_v<std::tuple<TT...>>,
		"The number of argument must equal to the number of element in the binding tuple variable");

	//TODO: capture `auto` as template argument in C++ 20 so it's less verbose
	auto convertOne = [i = static_cast<size_t>(0u), &rx_arg](auto& dst) mutable -> void {
		dst = rx_arg[i++].to<std::remove_reference_t<decltype(dst)>>();
	};
	//iterate over each element in the tuple
	std::apply([&convertOne](auto&... arg) { (convertOne(arg), ...); }, *this->Variable);
}

#undef OPTION_CTOR
#undef OPTION_CONVERT
/* -------------------------------------------------- result ----------------------------------------------------- */

inline const NAMESPACE_CMD_NAME::STPInternal::STPBaseCommand& NAMESPACE_CMD_NAME::STPParseResult::commandBranch() const noexcept {
	return *this->HelpData.CommandPath.back();
}

/* --------------------------------------------------------------------------------------------------------------- */

#undef NAMESPACE_CMD_NAME

#endif//_STP_COMMAND_LINE_PARSER_H_