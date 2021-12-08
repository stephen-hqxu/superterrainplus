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
		const STPSmartShaderObject Shader;

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
		 * @brief Attach source code to the current shader manager and compile. Previously attached source code will be removed.
		 * @param source A single string that contains all GLSL source code.
		 * The source must corresponds the type of the shader.
		 * @return Compilation log, if any.
		 * If compilation fails, exception is thrown with error log.
		*/
		const std::string& operator()(const std::string&);

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