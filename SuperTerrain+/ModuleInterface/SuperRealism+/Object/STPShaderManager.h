#pragma once
#ifndef _STP_SHADER_MANAGER_H_
#define _STP_SHADER_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

//System
#include <string>
#include <vector>

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
		bool Valid;

		/**
		 * @brief STPShaderDeleter calls glDeleteShader to remove a shader.
		*/
		struct STP_REALISM_API STPShaderDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPShaderDeleter>> STPSmartShaderObject;
		//An OpenGL shader object
		const STPSmartShaderObject Shader;

	public:

		//Type of the shader currently managed by the shader manager.
		const STPOpenGL::STPenum Type;

		/**
		 * @brief Initialise a new shader manager.
		 * @param source A single string that contains all GLSL source code.
		 * @param type The type of this shader.
		*/
		STPShaderManager(const std::string&, STPOpenGL::STPenum);

		/**
		 * @brief Intialise a new shader manager.
		 * @param source A vector of string, where each element is one line of source code.
		 * Each line must be null-terminated.
		 * @param type The type of this shader.
		*/
		STPShaderManager(const std::vector<const char*>&, STPOpenGL::STPenum);

		STPShaderManager(const STPShaderManager&) = delete;

		STPShaderManager(STPShaderManager&&) noexcept = default;

		STPShaderManager& operator=(const STPShaderManager&) = delete;

		STPShaderManager& operator=(STPShaderManager&&) noexcept = default;

		~STPShaderManager() = default;

		/**
		 * @brief Get the compilation log from the shader object.
		 * @return The compilation log.
		 * If there is no compilation log, nothing is returned.
		*/
		const std::string& getLog() const;

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