#include <SuperRealism+/Object/STPShaderManager.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//System
#include <sstream>

using std::string;
using std::istringstream;
using std::ostringstream;
using std::vector;

using std::endl;

using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

void STPShaderManager::STPShaderDeleter::operator()(STPOpenGL::STPuint shader) const {
	glDeleteShader(shader);
}

STPShaderManager::STPShaderManager(STPOpenGL::STPenum type) : Shader(glCreateShader(type)), Type(type) {
	
}

void STPShaderManager::cache(const string& source) {
	this->SourceCache = source;
}

unsigned int STPShaderManager::defineMacro(const STPMacroValueDictionary& dictionary) {
	if (this->SourceCache.empty()) {
		//do nothing for empty string
		return 0u;
	}
	constexpr static char DefineIdentifier[] = "#define";
	constexpr static char MacroIdentifier[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_";

	unsigned int macroReplaced = 0u;
	istringstream original_src(this->SourceCache);
	ostringstream output_src;
	string line, macroName;
	//read line by line
	while (std::getline(original_src, line)) {
		if (line.rfind(DefineIdentifier, 0ull) == 0ull) {
			//a define directive is found
			//for accuracy, we only care about #define starting at the first line
			const size_t macroStart = line.find(' ', sizeof(DefineIdentifier) / sizeof(char) - 1ull) + 1ull;
			if (macroStart != string::npos) {
				//get macro name
				const size_t macroEnd = line.find_first_not_of(MacroIdentifier, macroStart);
				//substr will clamp the size so no worries about npos
				macroName = line.substr(macroStart, macroEnd - macroStart);

				//get macro value
				auto dict_it = dictionary.find(macroName);
				if (dict_it != dictionary.cend()) {
					//found, compose new line
					output_src << DefineIdentifier << ' ' << macroName << ' ' << dict_it->second << endl;

					macroReplaced++;
					continue;
				}
			}
		}
		
		//insert line into a new cache
		output_src << line << endl;
	}
	//copy stream to cache
	this->SourceCache = output_src.str();

	return macroReplaced;
}

const string& STPShaderManager::operator()(const string& source, const STPShaderIncludePath& include) {
	//attach source code to the shader
	//std::string makes sure string is null-terminated, so we can pass NULL as the code length
	const char* const sourceArray = source.c_str();
	const GLint sourceLength = static_cast<GLint>(source.size());
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

		glCompileShaderIncludeARB(this->Shader.get(), pathCount, pathStr.data(), pathLength.data());
	}
	//retrieve any log
	GLint logLength, status;
	glGetShaderiv(this->Shader.get(), GL_INFO_LOG_LENGTH, &logLength);
	glGetShaderiv(this->Shader.get(), GL_COMPILE_STATUS, &status);

	//store information
	this->Valid = status == GL_TRUE ? true : false;
	if (logLength > 0) {
		//shader compilation has log
		this->Log.resize(logLength);
		glGetShaderInfoLog(this->Shader.get(), logLength, NULL, this->Log.data());
	}
	else {
		//clear old compilation old, leaving no log at current
		this->Log.clear();
	}

	if (!this->Valid) {
		//compilation error
		throw STPException::STPGLError(this->Log.c_str());
	}
	return this->lastLog();
}

const string& STPShaderManager::operator()(const STPShaderIncludePath& include) {
	if (this->SourceCache.empty()) {
		throw STPException::STPMemoryError("There is no source code cached to this shader manager");
	}

	return  this->operator()(this->SourceCache, include);
}

const string& STPShaderManager::lastLog() const {
	return this->Log;
}

STPShaderManager::operator bool() const {
	return this->Valid;
}

SuperTerrainPlus::STPOpenGL::STPuint STPShaderManager::operator*() const {
	return this->Shader.get();
}