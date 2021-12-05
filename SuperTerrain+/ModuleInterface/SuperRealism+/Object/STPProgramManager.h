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
		//An array of program parameters to be applied before program is linked.
		typedef std::list<std::pair<STPOpenGL::STPenum, STPOpenGL::STPint>> STPProgramParameteri;

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
		bool Linked, Valid;

	public:

		/**
		 * @brief Intialise a STPProgramManager.
		 * @param shader_group A pointer to an array of shaders to be attached to this program.
		 * According to OpenGL specification, shader can be deleted safely after it has been attached to ba program, 
		 * and OpenGL will handle the rests.
		*/
		STPProgramManager(const STPShaderGroup&, const STPProgramParameteri& = { });

		STPProgramManager(const STPProgramManager&) = delete;

		STPProgramManager(STPProgramManager&&) noexcept = default;

		STPProgramManager& operator=(const STPProgramManager&) = delete;

		STPProgramManager& operator=(STPProgramManager&&) noexcept = default;

		~STPProgramManager() = default;

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
		 * @brief Get the log from the program object.
		 * @param log_type The type of the log.
		 * @return The log with specified type.
		 * If there is no such log, nothing is returned.
		*/
		const std::string& getLog(STPLogType) const;

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