#pragma once
#ifndef _STP_SHADER_MANAGER_H_
#define _STP_SHADER_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Compatibility
#include <SuperTerrain+/STPOpenGL.h>

//System
#include <string>
#include <string_view>
#include <vector>
#include <optional>

#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShaderManager is a smart manager to OpenGL shader. It automatically loads GLSL source code and attach it to a managed shader object.
	 * Following OpenGL specification, as soon as a shader is attached to a program, it can be deleted, and OpenGL should handle the rest automatically.
	*/
	class STP_REALISM_API STPShaderManager {
	private:

		//Any log that comes from the compilation.
		std::unique_ptr<char[]> Log;
		//Indication of compilation status
		bool Valid;

	public:

		//An OpenGL shader object
		const STPOpenGL::STPuint Shader;
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

		STPShaderManager(STPShaderManager&&) = delete;

		STPShaderManager& operator=(const STPShaderManager&) = delete;

		STPShaderManager& operator=(STPShaderManager&&) = delete;

		~STPShaderManager();

		/**
		 * @brief Get the compilation log from the shader object.
		 * @return The compilation log.
		 * If there is no compilation log, nothing is returned.
		*/
		std::optional<std::string_view> getLog() const;

		/**
		 * @brief Check if the current shader manager is valid.
		*/
		explicit operator bool() const;

	};

}
#endif//_STP_SHADER_MANAGER_H_