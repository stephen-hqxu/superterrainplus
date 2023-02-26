//TEMPLATE DEFINITION FOR COMMAND LINE PARSER
#ifdef _STP_COMMAND_LINE_PARSER_H_

#include <algorithm>
#include <limits>
#include <optional>

#define NAMESPACE_CMD_NAME SuperTerrainPlus::STPAlgorithm::STPCommandLineParser

/* ---------------------------- count ---------------------------- */
#define COUNT_REQ_NAME NAMESPACE_CMD_NAME::STPInternal::STPCountRequirement

constexpr COUNT_REQ_NAME::STPCountRequirement() noexcept : Min(0u), Max(0u) {

}

constexpr void COUNT_REQ_NAME::set(const size_t num) noexcept {
	this->Min = num;
	this->Max = num;
}

constexpr void COUNT_REQ_NAME::unlimitedMax() noexcept {
	this->Max = std::numeric_limits<size_t>::max();
}

constexpr bool COUNT_REQ_NAME::isMaxUnlimited() const noexcept {
	return this->Max == std::numeric_limits<size_t>::max();
}

#undef COUNT_REQ_NAME
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
#define BASE_COMMAND_NAME NAMESPACE_CMD_NAME::STPInternal::STPBaseCommand

inline bool BASE_COMMAND_NAME::isGroup() const noexcept {
	return this->IsGroup;
}

inline bool BASE_COMMAND_NAME::isSubcommand() const noexcept {
	return !this->IsGroup;
}

#undef BASE_COMMAND_NAME

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
#define BASE_OPTION_NAME NAMESPACE_CMD_NAME::STPInternal::STPBaseOption

inline bool BASE_OPTION_NAME::isPositional() const noexcept {
	return this->PositionalPrecedence > 0u;
}

inline bool BASE_OPTION_NAME::supportDelimiter() const noexcept {
	return this->Delimiter != '\0';
}

inline bool BASE_OPTION_NAME::used() const noexcept {
	return this->Result.IsUsed;
}

#undef BASE_OPTION_NAME

#define OPTION_CONVERT(TEMP, CONV) inline void NAMESPACE_CMD_NAME::STPOption<TEMP, CONV>::convert([[maybe_unused]] const STPReceivedArgument& rx_arg) const
#define OPTION_CTOR(TEMP, CONV) \
inline NAMESPACE_CMD_NAME::STPOption<TEMP, CONV>::STPOption(TEMP& variable, Conv&& converter) : \
	Converter(std::forward<Conv>(converter)), Variable(&variable)

#define OPTION_SPEC_CONVERT(TEMP) template<class Conv> OPTION_CONVERT(TEMP, Conv)
#define OPTION_SPEC_CTOR(TEMP) template<class Conv> OPTION_CTOR(TEMP, Conv)

template<class BT, class Conv>
OPTION_CTOR(BT, Conv) { }

template<class BT, class Conv>
OPTION_CONVERT(BT, Conv) {
	if (rx_arg.empty()) {
		//nothing to be converted
		return;
	}

	BT variable {};
	if (this->Converter(std::make_pair(rx_arg.cbegin(), rx_arg.cend()), variable) != rx_arg.size()) {
		STPInternal::STPBaseOption::throwConversionError("All arguments are expected to be consumed by the converter", rx_arg);
	}

	*this->Variable = std::move(variable);
}

OPTION_SPEC_CONVERT(void) {
	//do nothing
}

OPTION_SPEC_CTOR(bool) { }

OPTION_SPEC_CONVERT(bool) {
	if (rx_arg.size() > 1u) {
		STPInternal::STPBaseOption::throwConversionError(
			"The boolean binding variable specialisation can only accept at most one argument", rx_arg);
	}

	bool variable;
	const size_t convertedCount = this->Converter(std::make_pair(rx_arg.cbegin(), rx_arg.cend()), variable);

	//if there is an argument, read directly from it
	//a flag does not have any argument, simply assign from its inferred value
	*this->Variable = convertedCount == 1u ? variable : this->InferredValue;
}

#undef OPTION_SPEC_CONVERT
#undef OPTION_SPEC_CTOR

#undef OPTION_CTOR
#undef OPTION_CONVERT
/* -------------------------------------------- argument converter ----------------------------------------------- */
#define ARG_CONV_NAME NAMESPACE_CMD_NAME::STPInternal::STPArgumentConverter
#define ARG_CONV_FUNC_DEF(TYPE) \
inline size_t ARG_CONV_NAME<TYPE>::operator()(const STPBaseOption::STPReceivedArgumentSpan& rx_arg, TYPE& var) const

#define ARG_CONV_SPEC(TEMP, ARG) template<TEMP> struct ARG_CONV_NAME<ARG> { \
	size_t operator()(const STPBaseOption::STPReceivedArgumentSpan&, ARG&) const; \
}; \
template<TEMP> \
ARG_CONV_FUNC_DEF(ARG)

//base case for conversion of a single fundamental type
//all specialisations will be recursive cases
template<typename T>
ARG_CONV_FUNC_DEF(T) {
	const auto [beg, end] = rx_arg;
	if (beg == end) {
		//we expect at least one argument
		return 0u;
	}

	//just use our string tool to convert the argument
	var = beg->to<T>();
	return 1u;
}

//a wrapper type of optional
ARG_CONV_SPEC(typename T, std::optional<T>) {
	//convert the inner type
	T inner {};
	if (const size_t converted = STPArgumentConverter<T> {}(rx_arg, inner);
		converted > 0u) {
		var.emplace(std::move(inner));
		return converted;
	}

	//clear optional if there is nothing converted
	var.reset();
	return 0u;
}

//a vector, so can take variable number of argument
ARG_CONV_SPEC(typename T, std::vector<T>) {
	auto [beg, end] = rx_arg;
	size_t convertedCount = 0u;

	var.clear();
	//try to convert each inner argument
	while (beg < end) {
		T inner {};
		const size_t converted = STPArgumentConverter<T> {}(std::make_pair(beg, end), inner);
		if (converted == 0u) {
			//cannot be converted
			break;
		}

		var.emplace_back(std::move(inner));
		beg += converted;
		convertedCount += converted;
	}
	return convertedCount;
}

//a tuple, it has a fixed size
ARG_CONV_SPEC(typename... T, std::tuple<T...>) {
	using std::move;
	
	size_t convertedCount = 0u;
	//return false to interrupt the loop
	//force copy the argument as non-const
	auto convertOne = [rx_arg = rx_arg, &convertedCount](auto& dst) mutable -> bool {
		//TODO: capture `auto` as template argument in C++ 20 so it's less verbose
		typedef std::decay_t<decltype(dst)> ArgT;

		//convert the current tuple element type
		auto& [beg, end] = rx_arg;
		ArgT argument {};
		//we enforce that every tuple element must be assigned with a converted argument
		const size_t converted = STPArgumentConverter<ArgT> {}(std::make_pair(beg, end), argument);
		if (converted == 0u) {
			return false;
		}

		dst = move(argument);
		beg += converted;
		convertedCount += converted;
		return true;
	};

	//we should maintain an origin copy to recover if conversion fails
	std::tuple<T...> result {};
	bool allConoverted = true;

	std::apply([&convertOne, &allConoverted](auto&... arg) { allConoverted = (convertOne(arg) && ...); }, result);
	//we require all elements in a tuple are converted
	if (!allConoverted) {
		return 0u;
	}
	//if conversion goes well, clear all original values and copy the result in
	var = move(result);
	return convertedCount;
}

#undef ARG_CONV_SPEC

#undef ARG_CONV_FUNC_DEF
#undef ARG_CONV_NAME
/* -------------------------------------------------- result ----------------------------------------------------- */

inline const NAMESPACE_CMD_NAME::STPInternal::STPBaseCommand& NAMESPACE_CMD_NAME::STPParseResult::commandBranch() const noexcept {
	return *this->HelpData.CommandPath.back();
}

/* --------------------------------------------------------------------------------------------------------------- */

#undef NAMESPACE_CMD_NAME

#endif//_STP_COMMAND_LINE_PARSER_H_