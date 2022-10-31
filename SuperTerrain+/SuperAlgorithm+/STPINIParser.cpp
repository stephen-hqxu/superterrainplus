#include <SuperAlgorithm+/Parser/INI/STPINIReader.h>
#include <SuperAlgorithm+/Parser/INI/STPINIWriter.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>

#include <optional>
#include <sstream>

#include <utility>

using std::optional;
using std::make_optional;
using std::nullopt;
using std::string_view;
using std::pair;
using std::make_pair;
using std::ostringstream;

using std::endl;

using namespace SuperTerrainPlus::STPAlgorithm;

/* ======================================= STPINIReader.h ====================================== */

class STPINIReader::STPINIReaderImpl {
private:

	//white space sequence identifier
	constexpr static string_view WhiteSpace = " \n\r\t\f\v";

	const string_view Str;
	//The current parsing line number
	size_t Line;
	//The character sequence referencing the string
	const char* Sequence;

	//Remove all leading white space
	inline static string_view ltrim(const string_view& s) {
		const size_t start = s.find_first_not_of(STPINIReaderImpl::WhiteSpace);
		return start == string_view::npos ? string_view() : s.substr(start);
	}

	//Remove all trailing white space
	inline static string_view rtrim(const string_view& s) {
		const size_t end = s.find_last_not_of(STPINIReaderImpl::WhiteSpace);
		return end == string_view::npos ? string_view() : s.substr(0, end + 1);
	}

	//Remove white space from both ends
	inline static string_view doubleTrim(const string_view& s) {
		return STPINIReaderImpl::rtrim(STPINIReaderImpl::ltrim(s));
	}

public:

	/**
	 * @brief Initialise an implementation of INI reader.
	 * @param str The pointer to the INI string.
	*/
	STPINIReaderImpl(const string_view& str) : Str(str), Line(0u), Sequence(this->Str.data()) {

	}

	STPINIReaderImpl(const STPINIReaderImpl&) = delete;

	STPINIReaderImpl(STPINIReaderImpl&&) = delete;

	STPINIReaderImpl& operator=(const STPINIReaderImpl&) = delete;

	STPINIReaderImpl& operator=(STPINIReaderImpl&&) = delete;

	~STPINIReaderImpl() = default;

	/**
	 * @brief Read the next line in the raw INI string sequence.
	 * @return The view of the line.
	 * If there is no more line available, return null.
	*/
	optional<string_view> getLine() {
		const char* const start = this->Sequence;
		if (*start == '\0') {
			//end of the string, return null
			return nullopt;
		}

		while (true) {
			switch (*this->Sequence) {
			case '\n':
				//for new line delimiter, discard this character and advance then return
				this->Sequence++;
				this->Line++;
				[[fallthrough]];
			case '\0':
				//for end of file, do not advance to the next character

				//TODO: C++20 allows creating a view from first and last pointer
				return STPINIReaderImpl::doubleTrim(string_view(start, this->Sequence - start));
			default:
				//advance
				this->Sequence++;
			}
		}
	}

	/**
	 * @brief Parse the line that contain section name.
	 * @param line - The current line of the INI.
	 * @return The name of that section, or empty view if there is a syntax error.
	*/
	inline string_view parseSection(string_view line) {
		//[section]
		//we only need to remove the surrounding bracket
		if (line.front() != '[' || line.back() != ']') {
			//incomplete bracket, syntax error
			return string_view();
		}
		return STPINIReaderImpl::doubleTrim(line.substr(1u, line.length() - 2u));
	}

	/**
	 * @brief Parse the current line into a property.
	 * @param line - The current line of the INI.
	 * @return The key-value pair of the current line, or null if there is a syntax error.
	*/
	inline optional<pair<string_view, string_view>> parseProperty(const string_view& line) {
		//key=value
		//we only need to find the location of the `=`
		const size_t loc = line.find_first_of('=');
		if (loc == string_view::npos) {
			//not found
			return nullopt;
		}
		//substring to get the key and value
		return make_pair(STPINIReaderImpl::doubleTrim(line.substr(0u, loc)), STPINIReaderImpl::doubleTrim(line.substr(loc + 1u)));
	}

	/**
	 * @brief Create an initial error message.
	 * @param ss The pointer to the output string stream.
	 * @param error_type User specified error type message.
	 * @return The input string stream.
	*/
	inline ostringstream& createInitialErrorMessage(ostringstream& ss, const char* error_type) {
		ss << "SuperTerrain+ INI Reader(" << this->Line << "):" << error_type << endl;
		return ss;
	}

};

STPINIReader::STPINIReader(const string_view& ini_str) {
	STPINIReaderImpl reader(ini_str);

	optional<string_view> line;
	//start with unnamed section
	STPINISectionView* current_sec = &this->addSection("");
	while ((line = reader.getLine())) {
		if (line->empty()) {
			//skip empty line
			continue;
		}

		//otherwise we check the starting point of the line to determine what we are going to do
		switch (line->front()) {
		case '[':
		{
			//section
			const string_view next_sec = reader.parseSection(*line);
			if (next_sec.empty()) {
				//syntax error
				ostringstream msg;
				reader.createInitialErrorMessage(msg, "invalid section")
					<< "Line value \'" << *line << "\' that is supposed to be a section cannot be parsed." << endl;
				
				throw STPException::STPInvalidSyntax(msg.str().c_str());
			}

			//start a new section
			current_sec = &this->addSection(next_sec);
		}
			break;

		case '#':
		case ';':
			//comment, skip
			break;

		default:
		{
			//property
			const auto next_prop = reader.parseProperty(*line);
			if (!next_prop.has_value()) {
				//syntax error
				ostringstream msg;
				reader.createInitialErrorMessage(msg, "invalid property")
					<< "Line value \'" << *line << "\' that is supposed to be a property cannot be parsed." << endl;

				throw STPException::STPInvalidSyntax(msg.str().c_str());
			}
			const auto& [key, value] = *next_prop;

			//note that if the key is duplicated, a new value will be written in anyway, such that the old value is discarded
			(*current_sec)[key] = STPINIStringView(value);
		}
			break;
		}
	}
}

inline STPINISectionView& STPINIReader::addSection(const string_view& sec_name) {
	return this->Data.try_emplace(sec_name).first->second;
}

const STPINIStorageView& STPINIReader::operator*() const {
	return this->Data;
}

/* ====================================== STPINIWriter.h ========================================= */

using std::string;

STPINIWriter::STPINIWriter(const STPINIStorageView& storage, STPWriterFlag flag) {
	//process flags
	static constexpr auto readFlag = [](STPWriterFlag op, STPWriterFlag against) constexpr -> bool {
		return (op & against) != 0u;
	};
	const bool control[3] = {
		readFlag(flag, STPINIWriter::SectionNewline),
		readFlag(flag, STPINIWriter::SpaceAroundAssignment),
		readFlag(flag, STPINIWriter::SpaceAroundSectionName)
	};
	const string_view equalMark = (control[1] ? " = " : "="),
		leftSqBracket = (control[2] ? "[ " : "["),
		rightSqBracket = (control[2] ? " ]" : "]");

	//prepare output
	ostringstream output;
	auto writeSection = [&output, &equalMark](const STPINISectionView& section) -> void {
		for (const auto& [key, value] : section) {
			output << key << equalMark << value << endl;
		}
	};

	//output unnamed section first, since unnamed section should start at first line
	auto it = storage.find("");
	if (it != storage.cend() && !it->second.empty()) {
		writeSection(it->second);
	}

	//iterate over the rest of sections
	for (const auto& [name, section] : storage) {
		if (name == "") {
			//do not output default section since we have done it.
			continue;
		}

		if (control[0]) {
			//new line between each section
			output << endl;
		}

		//output section declare line
		output << leftSqBracket << name << rightSqBracket << endl;

		//iterate over the properties
		writeSection(section);
	}

	this->Data = output.str();
	this->Data.shrink_to_fit();
}

const string& STPINIWriter::operator*() const {
	return this->Data;
}