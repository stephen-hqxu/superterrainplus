#pragma once
#ifndef _STP_PROGRAM_MANAGER_H_
#define _STP_PROGRAM_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>
//Shader
#include "STPShaderManager.h"

//Container
#include <list>
#include <tuple>
#include <unordered_map>

#include <functional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPProgramManager is a smart manager to OpenGL shader program.
	 * It links with multiple shaders into a single program.
	*/
	class STP_REALISM_API STPProgramManager {
	public:

		//All shader manager attached to this program.
		typedef std::list<const STPShaderManager*> STPShaderGroup;

		/**
		 * @brief STPLogType specifies the type of the log retrieved from a program object.
		*/
		enum class STPLogType : unsigned char {
			Link = 0x00u,
			Validation = 0xFFu
		};

	private:

		/**
		 * @brief STPProgramDeleter calls glDeleteProgram to remove a program.
		*/
		struct STP_REALISM_API STPProgramDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPProgramDeleter>> STPSmartProgram;
		//A shader program
		const STPSmartProgram Program;

		//Program linking log
		std::string LinkLog, ValidationLog;
		//A value to denotes if the program is linked and validated.
		bool Linked = false, Valid = false;

		std::unordered_map<STPOpenGL::STPenum, STPOpenGL::STPuint> AttachedShader;

	public:

		/**
		 * @brief Intialise a STPProgramManager.
		*/
		STPProgramManager();

		STPProgramManager(const STPProgramManager&) = delete;

		STPProgramManager(STPProgramManager&&) noexcept = default;

		STPProgramManager& operator=(const STPProgramManager&) = delete;

		STPProgramManager& operator=(STPProgramManager&&) noexcept = default;

		~STPProgramManager() = default;

		/**
		 * @brief Attach a new shaders to the current program.
		 * @param shader_group A pointer to an array of shaders to be attached to this program.
		 * @return The pointer to the current program manager for chaining.
		 * If shader type repeats, or shader fails to compile, exception is thrown.
		*/
		STPProgramManager& attach(const STPShaderGroup&);

		/**
		 * @brief Detatch a shader from the current program.
		 * @param type The type of the shader to be detached.
		 * @return True if it has been detached. False if no type of this shader is found.
		*/
		bool detach(STPOpenGL::STPenum);

		/**
		 * @brief Detach all shaders from the current program.
		 * This does not reset program parameters, however.
		*/
		void clear();

		/**
		 * @brief Finalise the shader program by linking all shaders.
		 * Any program parameters setting must be done before linking.
		 * Linkage may fail, so make sure to call lastLog() to get error message.
		*/
		void finalise();

		/**
		 * @brief Get the uniform location for a uniform in the current program.
		 * @param uni The name of the uniform.
		 * @return The uniform location in this program.
		*/
		STPOpenGL::STPint uniformLocation(const char*) const;

		/**
		 * @brief Perform glProgramUniform... operation for the current program object.
		 * @tparam Uni The uniform function.
		 * @tparam ...Arg Argument to be passed to the function.
		 * @param uniform_function A uniform function to be executed. It must be a valid glProgramUniform... function.
		 * @param uni The name of the uniform.
		 * @param ...args The arguments for the function.
		 * @return The pointer to *this* for chaining.
		*/
		template<typename Uni, typename... Arg>
		const STPProgramManager& uniform(Uni&&, const char*, Arg&&...) const;

		/**
		 * @brief Get the log from the last program object linking.
		 * @param log_type The type of the log.
		 * @return The log with specified type.
		 * If there is no such log, empty string is returned.
		*/
		const std::string& lastLog(STPLogType) const;

		/**
		 * @brief Check if the program is linked and validated such that it can be used.
		*/
		explicit operator bool() const;

		/**
		 * @brief Get the underlying program object.
		 * @return The program object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Use the current program object to make it active.
		*/
		void use() const;

		/**
		 * @brief Clear all active used program and reset it to default.
		*/
		static void unuse();

	};

}
#include "STPProgramManager.inl"
#endif//_STP_PROGRAM_MANAGER_H_