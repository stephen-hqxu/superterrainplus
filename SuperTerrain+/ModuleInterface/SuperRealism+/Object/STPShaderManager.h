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
	public:

		/**
		 * @brief STPShaderSource contains information about a shader source code and allow 
		 * pre-process the code before compilation.
		*/
		class STP_REALISM_API STPShaderSource {
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
				 * @tparam T The type of the value. This type must be convertable to string defined by standard library function to_string().
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

				STPShaderIncludePath() = default;;

				~STPShaderIncludePath() = default;

				/**
				 * @brief Insert an include path.
				 * @param path The string literal pathname. The content of this pointer will be copied.
				 * @return The pointer to the current include path manager for chaining.
				*/
				STPShaderIncludePath& operator[](const char*);

			};

		private:

			//Attached source code.
			std::string Cache;

		public:

			//A list of string that contains include paths of a shader.
			STPShaderIncludePath Include;

			/**
			 * @brief Create a new shader source class.
			 * @param source The pointer to the source code. It will be copied.
			*/
			STPShaderSource(const std::string&);

			STPShaderSource(const STPShaderSource&) = default;

			STPShaderSource(STPShaderSource&&) noexcept = default;

			STPShaderSource& operator=(const STPShaderSource&) = default;

			STPShaderSource& operator=(STPShaderSource&&) noexcept = default;

			~STPShaderSource() = default;

			/**
			 * @brief Get the source cached in the shader source instance.
			 * @return The pointer to the source.
			*/
			const std::string& operator*() const;

			/**
			 * @brief Assign macros with values in the cached source code.
			 * If macro has been defined with values, it will be replaced with the new value.
			 * @param dictionary A lookup table that maps macro names to values.
			 * @return The number of macro being assigned with values.
			*/
			unsigned int define(const STPMacroValueDictionary&);

		};

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

	public:

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
		 * @brief Initialise the shader manager for the current context.
		 * Specifically, it flushes all system shader include code (i.e., shader code used by the rendering engine) to GL include tree.
		 * Each context only needs to be initialised once.
		*/
		static void initialise();

		/**
		 * @brief Add a shader source to GL include file tree.
		 * @param name The name of the include shader.
		 * @param source The string, i.e., source, of the shader.
		 * @return True if the shader has been added, false if the same name has been included previously.
		*/
		static bool include(const std::string&, const std::string&);

		/**
		 * @brief Remove a shader source from GL include file tree.
		 * GL will throw error if name is not valid.
		 * @param name The name of the include shader.
		*/
		static void uninclude(const std::string&);

		/**
		 * @brief Attach source code to the current shader manager and compile. Previously attached source code will be removed.
		 * @param source The pointer to the shader source manager.
		 * @return Compilation log, if any.
		 * If compilation fails, exception is thrown with error log.
		*/
		const std::string& operator()(const STPShaderSource&);

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
#include "STPShaderManager.inl"
#endif//_STP_SHADER_MANAGER_H_