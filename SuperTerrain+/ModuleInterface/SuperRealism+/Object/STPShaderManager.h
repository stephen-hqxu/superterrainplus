#pragma once
#ifndef _STP_SHADER_MANAGER_H_
#define _STP_SHADER_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

//System
#include <string>
#include <vector>
#include <unordered_map>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShaderManager is a smart manager to OpenGL shader. It automatically loads GLSL source code and attach it to a managed shader object.
	 * Following OpenGL specification, as soon as a shader is attached to a program, it can be deleted, and OpenGL should handle the rest automatically.
	*/
	class STP_REALISM_API STPShaderManager {
	private:

		//Any log that comes from the compilation.
		std::string Log;
		//Indication of compilation status
		bool Valid = false;

		/**
		 * @brief STPShaderDeleter calls glDeleteShader to remove a shader.
		*/
		struct STP_REALISM_API STPShaderDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPShaderDeleter>> STPSmartShaderObject;
		//An OpenGL shader object
		STPSmartShaderObject Shader;

		//Cached shader source code
		std::string SourceCache;

	public:

		//An array of string indicating the include path of a shader.
		typedef std::list<std::string> STPShaderIncludePath;
		//A dictionary to assign value to a macro
		typedef std::unordered_map<std::string, std::string> STPMacroValueDictionary;

		//Type of the shader currently managed by the shader manager.
		const STPOpenGL::STPenum Type;

		/**
		 * @brief Initialise a new shader manager.
		 * @param type The type of this shader.
		*/
		STPShaderManager(STPOpenGL::STPenum);

		STPShaderManager(const STPShaderManager&) = delete;

		STPShaderManager(STPShaderManager&&) noexcept = default;

		STPShaderManager& operator=(const STPShaderManager&) = delete;

		STPShaderManager& operator=(STPShaderManager&&) noexcept = default;

		~STPShaderManager() = default;

		/**
		 * @brief Add a shader source to the internal shader include cache.
		 * The shader source is only cached to an internal dictionary, it will be flushed to OpenGL during shader compilation when 
		 * the include path is being used but no existing named string is found in the current GL context, 
		 * according to ARB_shading_language_include sprcification.
		 * This effectively allows having the same shader include system among different GL contexts.
		 * @param name The name of the include shader.
		 * @param source The string, i.e., source, of the shader.
		 * @return True if the shader has been added, false if the same name has been included previously.
		*/
		static bool include(const std::string&, const std::string&);

		/**
		 * @brief Remove a shader source from the internal shader cache. This will not remove named string from the GL context.
		 * @param name The name of the include shader.
		 * @return True if the shader has been removed.
		*/
		static bool uninclude(const std::string&);

		/**
		 * @brief Cache a shader source code into the shader manager.
		 * Previously cached source is abandoned.
		 * Caching source code allows shader manager to perform pre-processing before compilation.
		 * @param source A pointer to a string that contains all GLSL source code.
		*/
		void cache(const std::string&);

		/**
		 * @brief Assign macros with values in the cached source code.
		 * If macro has been defined with values, it will be replaced with the new value.
		 * @param dictionary A lookup table that maps macro names to values.
		 * @return The number of macro being assigned with values.
		*/
		unsigned int defineMacro(const STPMacroValueDictionary&);

		/**
		 * @brief Attach source code to the current shader manager and compile. Previously attached source code will be removed.
		 * @param source A single string that contains all GLSL source code.
		 * The source must corresponds the type of the shader.
		 * @param include An array of shader include path to the virtual include directory provided by OpenGL.
		 * The path must either be a valid name of a string source either submitted to GL shader include system manually, 
		 * or has been included into the internal shader include cache.
		 * In case both GL shader system and internal cache has shader source pointed by the same name, 
		 * GL shader system will be used first.
		 * @return Compilation log, if any.
		 * If compilation fails, exception is thrown with error log.
		*/
		const std::string& operator()(const std::string&, const STPShaderIncludePath& = { });

		/**
		 * @brief Compile cached source code to the current shader manager. Previous compilation will be removed.
		 * Exception is thrown if there is no cache being attached previously.
		 * @param include An array of shader include path.
		 * @return Compilation log.
		*/
		const std::string& operator()(const STPShaderIncludePath& = { });

		/**
		 * @brief Get the compilation log from the last shader object compilation.
		 * @return The compilation log from last time.
		 * If there is no compilation log, nothing is returned.
		*/
		const std::string& lastLog() const;

		/**
		 * @brief Check if the current shader manager is valid.
		*/
		explicit operator bool() const;

		/**
		 * @brief Get the underlying shader object.
		 * @return The shader object.
		*/
		STPOpenGL::STPuint operator*() const;

	};

}
#endif//_STP_SHADER_MANAGER_H_