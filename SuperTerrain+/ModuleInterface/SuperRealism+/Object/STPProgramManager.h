#pragma once
#ifndef _STP_PROGRAM_MANAGER_H_
#define _STP_PROGRAM_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL
#include "STPNullableObject.hpp"
//Shader
#include "STPShaderManager.h"

//GLM
#include <glm/vec3.hpp>

//Container
#include <tuple>
#include <unordered_map>

#include <functional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPProgramManager is a smart manager to OpenGL shader program.
	 * It links with multiple shaders into a single program.
	*/
	class STP_REALISM_API STPProgramManager {
	private:

		/**
		 * @brief STPProgramDeleter calls glDeleteProgram to remove a program.
		*/
		struct STP_REALISM_API STPProgramDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef STPSmartGLuintObject<STPProgramDeleter> STPSmartProgram;
		//A shader program
		STPSmartProgram Program;

		//Indicate if the current program is used as a compute shader.
		bool isComputeProgram;

		//Use shader type as key, find the shader reference number
		typedef std::unordered_map<STPOpenGL::STPenum, STPOpenGL::STPuint> STPShaderDatabase;
		STPShaderDatabase AttachedShader;

		/**
		 * @brief Detach a shader by the shader database iterator.
		 * @param it The iterator to the detaching shader.
		 * The iterator will be erased from the shader database after the function has returned.
		 * @return The iterator to the following element.
		*/
		STPShaderDatabase::iterator detachByIterator(STPShaderDatabase::iterator);

	public:

		/**
		 * @brief Initialise a STPProgramManager.
		*/
		STPProgramManager();

		STPProgramManager(const STPProgramManager&) = delete;

		STPProgramManager(STPProgramManager&&) noexcept = default;

		STPProgramManager& operator=(const STPProgramManager&) = delete;

		STPProgramManager& operator=(STPProgramManager&&) noexcept = default;

		~STPProgramManager() = default;

		/**
		 * @brief Attach a new shaders to the current program.
		 * @param shader A pointer to a shader to be attached to this program.
		 * @return The pointer to the current program manager for chaining.
		 * If shader type repeats, or shader fails to compile, exception is thrown.
		*/
		STPProgramManager& attach(const STPShaderManager&);

		/**
		 * @brief Detach a shader from the current program.
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
		 * @brief Flag the current program as a separable program, which can be used in program pipeline.
		 * @param separable True to indicate the new separable status of this program.
		*/
		void separable(bool);

		/**
		 * @brief Finalise the shader program by linking all shaders.
		 * Any program parameters setting must be done before linking.
		 * Linkage may fail and exception is thrown.
		 * Log generated during program linkage will be reflected to the log handler.
		 * @see STPLogHandler
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
		STPProgramManager& uniform(Uni&&, const char*, Arg&&...);

		/**
		 * @brief Perform glProgramUniform... operation for the current program object using an explicit uniform location.
		 * @tparam Uni The uniform function.
		 * @tparam ...Arg Argument to be passed to the function.
		 * @param uniform_function A uniform function to be executed. Must be a valid glProgramUniform... function.
		 * @param location The location of the uniform. No operation will be performed if the uniform location is indicated as not found.
		 * @param ...args The arguments for the function.
		 * @return The pointer to the current instance for chaining.
		*/
		template<typename Uni, typename... Arg>
		STPProgramManager& uniform(Uni&&, STPOpenGL::STPint, Arg&&...);

		/**
		 * @brief Query the local work group size of the compute program as specified by its input layout qualifier(s).
		 * If the program is not a compute program, exception is thrown.
		 * @return A vector of 3 integers containing the local work-group size
		*/
		glm::ivec3 workgroupSize() const;

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