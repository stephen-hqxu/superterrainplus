#include <SuperRealism+/Object/STPShaderManager.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperRealism+/STPRealismInfo.h>
//Log Output
#include <SuperRealism+/Utility/STPLogHandler.hpp>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//System
#include <array>
#include <sstream>
#include <string_view>

using std::array;
using std::string;
using std::string_view;
using std::istringstream;
using std::ostringstream;
using std::vector;

using std::endl;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

constexpr static array<string_view, 8ull> mShaderIncludeRegistry = {
	"/Common/STPAnimatedWave.glsl",
	"/Common/STPAtmosphericScattering.glsl",
	"/Common/STPCameraInformation.glsl",
	"/Common/STPGeometryBufferWriter.glsl",
	"/Common/STPLightSpaceInformation.glsl",
	"/Common/STPMaterialRegistry.glsl",
	"/Common/STPNullPointer.glsl",
	"/Common/STPSeparableShaderPredefine.glsl"
};

void STPShaderManager::STPShaderDeleter::operator()(STPOpenGL::STPuint shader) const {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderSource::STPShaderIncludePath& STPShaderManager::STPShaderSource::STPShaderIncludePath::operator[](const char* path) {
	this->Pathname.emplace_back(path);
	return *this;
}

STPShaderManager::STPShaderSource::STPShaderSource(const string& name, const string& source) : SourceName(name), Cache(source) {

}

const string& STPShaderManager::STPShaderSource::operator*() const {
	return this->Cache;
}

unsigned int STPShaderManager::STPShaderSource::define(const STPMacroValueDictionary& dictionary) {
	if (this->Cache.empty()) {
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
	istringstream original_src(this->Cache);
	ostringstream output_src;
	string line;
	//read line by line
	while (std::getline(original_src, line)) {
		//remove leading white space at the beginning of the define directive
		if (string_view lineView(ltrim(line));
			lineView.rfind(DefineIdentifier, 0ull) == 0ull) {
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
	this->Cache = output_src.str();

	return macroReplaced;
}

inline const string& STPShaderManager::STPShaderSource::getName() const {
	return this->SourceName;
}

STPShaderManager::STPShaderManager(STPOpenGL::STPenum type) : Shader(glCreateShader(type)), Type(type) {
	
}

inline static bool includeImpl(const char* name, size_t nameLen, const string& source) {
	//check if path exists as named string
	if (!glIsNamedStringARB(static_cast<GLint>(nameLen), name)) {
		//try to add the named string to GL virtual include system
		glNamedStringARB(GL_SHADER_INCLUDE_ARB, static_cast<GLint>(nameLen), name, static_cast<GLint>(source.size()), source.c_str());
		return true;
	}
	return false;
}

void STPShaderManager::initialise() {
	//load source code
	for (const auto& path : mShaderIncludeRegistry) {
		using namespace SuperTerrainPlus;

		ostringstream filename;
		filename << SuperRealismPlus_ShaderPath << path;
		includeImpl(path.data(), path.length(), *STPFile(filename.str().c_str()));
	}
}

bool STPShaderManager::include(const string& name, const string& source) {
	return includeImpl(name.c_str(), name.length(), source);
}

void STPShaderManager::uninclude(const string& name) {
	glDeleteNamedStringARB(static_cast<GLint>(name.size()), name.c_str());
}

void STPShaderManager::operator()(const STPShaderSource& source) {
	const string& src_str = *source;
	const auto& include = source.Include.Pathname;

	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	const char* const sourceArray = src_str.c_str();
	const GLint sourceLength = static_cast<GLint>(src_str.size());
	glShaderSource(this->Shader.get(), 1, &sourceArray, &sourceLength);

	//try to compile it
	if (include.empty()) {
		glCompileShader(this->Shader.get());
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

		glCompileShaderIncludeARB(this->Shader.get(), static_cast<GLsizei>(pathCount), pathStr.data(), pathLength.data());
	}
	//retrieve any log
	GLint logLength, status;
	glGetShaderiv(this->Shader.get(), GL_INFO_LOG_LENGTH, &logLength);
	glGetShaderiv(this->Shader.get(), GL_COMPILE_STATUS, &status);

	//store information
	string log;
	const bool valid = status == GL_TRUE ? true : false;
	if (logLength > 0) {
		//shader compilation has log
		log.resize(logLength);
		glGetShaderInfoLog(this->Shader.get(), logLength, NULL, log.data());

		const string& name = source.getName();
		if (!name.empty()) {
			ostringstream identifier;
			identifier << name << endl;

			//print a horizontal bar
			for (int i = 0; i < name.length(); i++) {
				identifier << '-';
			}
			identifier << endl;

			//append the source name to the log
			log.insert(0, identifier.str());
		}
	}

	if (!valid) {
		//compilation error
		throw STPException::STPGLError(log.c_str());
	}

	//write log
	STPLogHandler::ActiveLogHandler->handle(std::move(log));
}

SuperTerrainPlus::STPOpenGL::STPuint STPShaderManager::operator*() const {
	return this->Shader.get();
}