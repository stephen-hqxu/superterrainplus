#include <SuperRealism+/Object/STPShaderManager.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>
#include <SuperRealism+/STPRealismInfo.h>
//Log Output
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//Error
#include <SuperTerrain+/Exception/API/STPGLError.h>

//GLAD
#include <glad/glad.h>

#include <string_view>
#include <algorithm>
//System IO
#include <filesystem>
#include <sstream>

using std::string;
using std::string_view;
using std::istringstream;
using std::ostringstream;
using std::vector;

using std::endl;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

/* STPLogHandler */

namespace {
	struct STPDefaultLogHandler : public STPLogHandler::STPLogHandlerSolution {
	public:

		void handle(string_view) override {
			//trivially do nothing
		}

	};
}

//The system's default log handler, prefix denotes `*l*og *h*andler`
static STPDefaultLogHandler lhDefaultLogHandler;
//The currently active log handler, might be defined by user
static STPLogHandler::STPLogHandlerSolution* lhActiveLogHandler = &lhDefaultLogHandler;

void STPLogHandler::set(STPLogHandlerSolution* const solution) {
	if (solution) {
		//set to user-defined
		lhActiveLogHandler = solution;
	} else {
		//nullptr, reset to default
		lhActiveLogHandler = &lhDefaultLogHandler;
	}
}

void STPLogHandler::handle(const string_view log) {
	lhActiveLogHandler->handle(log);
}

/* STPShaderManager */

void STPShaderManager::STPDetail::STPShaderDeleter::operator()(const STPOpenGL::STPuint shader) const noexcept {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderSource::STPShaderIncludePath&
	STPShaderManager::STPShaderSource::STPShaderIncludePath::operator[](const char* const path) {
	this->Pathname.emplace_back(path);
	return *this;
}

STPShaderManager::STPShaderSource::STPShaderSource(string&& name, string&& source) :
	SourceName(std::move(name)), Source(std::move(source)) {

}

unsigned int STPShaderManager::STPShaderSource::define(const STPMacroValueDictionary& dictionary) {
	if (this->Source.empty()) {
		//do nothing for empty string
		return 0u;
	}
	constexpr static string_view DefineIdentifier = "#define",
		WhiteSpace = " \n\r\t\f\v";
	//remove all leading white space
	constexpr static auto ltrim = [](const string_view& str) constexpr -> string_view {
		const size_t start = str.find_first_not_of(WhiteSpace);
		return start == string_view::npos ? string_view() : str.substr(start);
	};

	unsigned int macroReplaced = 0u;
	istringstream original_src(this->Source);
	ostringstream output_src;
	string line;
	//read line by line
	while (std::getline(original_src, line)) {
		//remove leading white space at the beginning of the define directive
		if (string_view lineView(ltrim(line));
			lineView.rfind(DefineIdentifier, 0u) == 0u) {
			//remove #define and all white space between #define and the macro name
			lineView = ltrim(lineView.substr(DefineIdentifier.length()));

			//get macro name
			const size_t macroEnd = lineView.find_first_of(' ');
			//substr will clamp the size so no worries about npos
			//only left with the actual name of macro
			lineView = ltrim(lineView.substr(0, macroEnd));

			//get macro value
			if (auto dict_it = dictionary.Macro.find(string(lineView));
				dict_it != dictionary.Macro.cend()) {
				//found, compose new line
				output_src << DefineIdentifier << ' ' << lineView << ' ' << dict_it->second << endl;

				macroReplaced++;
				continue;
			}
		}

		//insert line into a new cache
		output_src << line << endl;
	}
	//copy stream to cache
	this->Source = output_src.str();

	return macroReplaced;
}

void STPShaderManager::initialise() {
	constexpr static auto ShaderIncludeRootFilename = STPStringUtility::concatCharArray(STPRealismInfo::ShaderPath, "/Common");

	//load shader include files by traversing all files in the include path
	using namespace std::filesystem;
	const path shaderRoot = path(STPRealismInfo::ShaderPath).make_preferred(),
		shaderIncludeRoot = path(ShaderIncludeRootFilename.data()).make_preferred(),
		shaderFileExt(".glsl");
	directory_iterator shader_include_dir(shaderIncludeRoot);
	for (const auto& entry : shader_include_dir) {
		const path& entry_path = entry.path();
		if (entry_path.extension() != shaderFileExt) {
			//don't care about non-GLSL shader source
			continue;
		}
		//load shader code into graphics context using the full filename
		const string shader_code = STPFile::read(entry_path.string().c_str());
		//we want to find the path relative to the shader root path
		string relative_entry_path = entry_path.lexically_proximate(shaderRoot).string();

		//normalise path separator so it is readable according to GL standard
		//On POSIX this should do nothing since our filename will not contain this special symbol.
		//On Windows this replaces back slashes with the forward ones.
		std::replace(relative_entry_path.begin(), relative_entry_path.end(), '\\', '/');
		relative_entry_path.reserve(relative_entry_path.length() + 1u);
		relative_entry_path.insert(0u, 1u, '/');

		glNamedStringARB(GL_SHADER_INCLUDE_ARB, static_cast<GLint>(relative_entry_path.length()),
			relative_entry_path.c_str(), static_cast<GLint>(shader_code.length()), shader_code.c_str());
	}
}

STPShaderManager::STPShader STPShaderManager::make(const STPOpenGL::STPenum type, const STPShaderSource& source) {
	const string& src_str = source.Source;
	const auto& include = source.Include.Pathname;
	STPShader shaderManaged = STPShader(glCreateShader(type));
	const GLuint shader = shaderManaged.get();
	
	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	const char* const sourceArray = src_str.c_str();
	const GLint sourceLength = static_cast<GLint>(src_str.size());
	glShaderSource(shader, 1, &sourceArray, &sourceLength);

	//try to compile it
	if (include.empty()) {
		glCompileShader(shader);
	}
	else {
		const size_t pathCount = include.size();
		//build the path information
		vector<const char*> pathStr;
		vector<GLint> pathLength;
		pathStr.reserve(pathCount);
		pathLength.reserve(pathCount);
		for (const auto& path : include) {
			pathStr.emplace_back(path.c_str());
			pathLength.emplace_back(static_cast<GLint>(path.size()));
		}

		glCompileShaderIncludeARB(shader, static_cast<GLsizei>(pathCount), pathStr.data(), pathLength.data());
	}
	//retrieve any log
	GLint logLength, status;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	//store information
	string log;
	const bool valid = status == GL_TRUE ? true : false;
	if (logLength > 0) {
		//shader compilation has log
		log.resize(logLength);
		glGetShaderInfoLog(shader, logLength, nullptr, log.data());

		const string& name = source.SourceName;
		if (!name.empty()) {
			ostringstream identifier;
			identifier << name << endl;

			//print a horizontal bar
			for (size_t i = 0u; i < name.length(); i++) {
				identifier << '-';
			}
			identifier << endl;

			//append the source name to the log
			log.insert(0, identifier.str());
		}
	}

	if (!valid) {
		//compilation error
		throw STP_GL_ERROR_CREATE(log);
	}

	//write log
	STPLogHandler::handle(log);

	return shaderManaged;
}

SuperTerrainPlus::STPOpenGL::STPint STPShaderManager::shaderType(const STPShader& shader) noexcept {
	GLint type;
	glGetShaderiv(shader.get(), GL_SHADER_TYPE, &type);
	return type;
}