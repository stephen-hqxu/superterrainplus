#pragma once
#ifndef _STP_SHADER_MANAGER_H_
#define _STP_SHADER_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include "STPNullableObject.hpp"

//Container
#include <string>
#include <list>
#include <vector>
#include <unordered_map>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShaderManager is a smart manager to OpenGL shader. It automatically loads GLSL source code and attach it to a managed shader object.
	 * Following OpenGL specification, as soon as a shader is attached to a program, it can be deleted, and OpenGL should handle the rest automatically.
	*/
	namespace STPShaderManager {

		//Internal implementation of the shader manager
		namespace STPShaderManagerDetail {

			/**
			 * @brief STPShaderDeleter calls glDeleteShader to remove a shader.
			*/
			struct STP_REALISM_API STPShaderDeleter {
			public:

				void operator()(STPOpenGL::STPuint) const noexcept;

			};

		}

		/**
		 * @brief STPShaderSource contains information about a shader source code and allow 
		 * pre-process the code before compilation.
		*/
		struct STP_REALISM_API STPShaderSource {
		public:

			/**
			 * @brief STPMacroValueDictionary is a dictionary to assign value to a macro.
			*/
			struct STPMacroValueDictionary {
			public:

				std::unordered_map<std::string, std::string> Macro;

				STPMacroValueDictionary() = default;

				~STPMacroValueDictionary() = default;

				/**
				 * @brief Insert a macro-value pair into the macro dictionary.
				 * @tparam T The type of the value. This type must be convertible to string defined by standard library function to_string().
				 * @param macro The name of the macro.
				 * @param value The value of the macro which will be converted to string.
				 * @return The pointer to the current dictionary for chaining.
				*/
				template<typename T>
				STPMacroValueDictionary& operator()(const std::string&, T&&);

			};

			/**
			 * @brief STPShaderIncludePath information about include paths used for a shader.
			 * This include path is used for speeding up compilation if the include file system is huge, and acts as a hint 
			 * to OpenGL to search in this list first before searching the entire virtual file system.
			 * The engine will flush the source code automatically to GL if it has been attached to the internal cache.
			 * Otherwise compilation will fail.
			*/
			struct STP_REALISM_API STPShaderIncludePath {
			public:

				std::list<std::string> Pathname;

				STPShaderIncludePath() = default;

				~STPShaderIncludePath() = default;

				/**
				 * @brief Insert an include path.
				 * @param path The string literal pathname. The content of this pointer will be copied.
				 * @return The pointer to the current include path manager for chaining.
				*/
				STPShaderIncludePath& operator[](const char*);

			};

			//The name of the source as an identifier in the compilation log;
			const std::string SourceName;
			//Attached source code.
			std::string Source;

			//A list of string that contains include paths of a shader.
			STPShaderIncludePath Include;

			/**
			 * @brief Create a new shader source class.
			 * @param name The rvalue reference to be moved to a source name as an identifier in the compilation log.
			 * If both the source name and the log is not empty, this source name will be prepended to the log output for easier debugging.
			 * @param source The rvalue reference to the source code. It will be moved.
			*/
			STPShaderSource(std::string&&, std::string&&);

			~STPShaderSource() = default;

			/**
			 * @brief Assign macros with values in the cached source code.
			 * If macro has been defined with values, it will be replaced with the new value.
			 * @param dictionary A lookup table that maps macro names to values.
			 * @return The number of macro being assigned with values.
			*/
			unsigned int define(const STPMacroValueDictionary&);

		};

		//A GL shader object with managed memory lifetime
		using STPShader = STPSmartGLuintObject<STPShaderManagerDetail::STPShaderDeleter>;

		/**
		 * @brief Initialise the shader manager for the current context.
		 * Specifically, it flushes all system shader include code (i.e., shader code used by the rendering engine) to GL include tree.
		 * Each context only needs to be initialised once.
		*/
		STP_REALISM_API void initialise();

		/**
		 * @brief Add a shader source to GL include file tree.
		 * @param name The name of the include shader.
		 * @param source The string, i.e., source, of the shader.
		 * @return True if the shader has been added, false if the same name has been included previously.
		*/
		STP_REALISM_API bool Addinclude(const std::string&, const std::string&) noexcept;

		/**
		 * @brief Remove a shader source from GL include file tree.
		 * GL will throw error if name is not valid.
		 * @param name The name of the include shader.
		*/
		STP_REALISM_API void Removeinclude(const std::string&) noexcept;

		/**
		 * @brief Attach source code to the current shader manager and compile. Previously attached source code will be removed.
		 * Shader compilation log will be reflected to the shader log handler.
		 * @param type The type of this shader.
		 * @param source The pointer to the shader source manager.
		*/
		STP_REALISM_API STPShader make(STPOpenGL::STPenum, const STPShaderSource&);

		/**
		 * @brief Get the type of the shader.
		 * @param shader The shader object.
		 * @return The type of the shader.
		*/
		STP_REALISM_API STPOpenGL::STPint shaderType(const STPShader&) noexcept;

	}

}
#include "STPShaderManager.inl"
#endif//_STP_SHADER_MANAGER_H_