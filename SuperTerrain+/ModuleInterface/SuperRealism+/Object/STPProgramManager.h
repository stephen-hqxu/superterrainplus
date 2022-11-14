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

#include <initializer_list>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPProgramManager is a smart manager to OpenGL shader program.
	 * It links with multiple shaders into a single program.
	*/
	class STP_REALISM_API STPProgramManager {
	public:

		/**
		 * @brief STPProgramStateManager un-uses a program and reset active program state to zero upon destruction.
		*/
		struct STP_REALISM_API STPProgramStateManager {
		public:

			STPProgramStateManager() = default;

			STPProgramStateManager(const STPProgramStateManager&) = delete;

			STPProgramStateManager(STPProgramStateManager&&) = delete;

			STPProgramStateManager& operator=(const STPProgramStateManager&) = delete;

			STPProgramStateManager& operator=(STPProgramStateManager&&) = delete;

			~STPProgramStateManager();

		};

	private:

		/**
		 * @brief STPProgramDeleter calls glDeleteProgram to remove a program.
		*/
		struct STP_REALISM_API STPProgramDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		typedef STPSmartGLuintObject<STPProgramDeleter> STPSmartProgram;
		//A shader program
		STPSmartProgram Program;

	public:

		/**
		 * @brief STPProgramParameter contains parameters to the program
		*/
		struct STPProgramParameter {
		public:

			//Flag the current program as a separable program, which can be used in program pipeline.
			bool Separable = false;

		};

		/**
		 * @brief Initialise an empty program manager.
		*/
		STPProgramManager() = default;

		/**
		 * @brief Initialise a program manager and link all shader together to form a complete program.
		 * @param shader_ptr An array of pointers, each to a shader to be linked.
		 * @param count The number of element in the array.
		 * For contiguous memory, this should be the number of shader object.
		 * For non-contiguous memory, this should be the number of pointer to shader objects.
		 * @param option Specifies the shader compiler option.
		*/
		STPProgramManager(const STPShaderManager::STPShader* const*, size_t, const STPProgramParameter* = nullptr);

		//Array of pointers, each to a shader object.
		//@see STPProgramManager
		STPProgramManager(std::initializer_list<const STPShaderManager::STPShader*>, const STPProgramParameter* = nullptr);

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
		STPOpenGL::STPint uniformLocation(const char*) const noexcept;

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
		STPProgramManager& uniform(Uni&&, const char*, Arg&&...) noexcept;

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
		STPProgramManager& uniform(Uni&&, STPOpenGL::STPint, Arg&&...) noexcept;

		/**
		 * @brief Query the local work group size of the compute program as specified by its input layout qualifier(s).
		 * @return A vector of 3 integers containing the local work-group size
		*/
		glm::ivec3 workgroupSize() const noexcept;

		/**
		 * @brief Get the underlying program object.
		 * @return The program object.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		/**
		 * @brief Use the current program object to make it active, with automatic program state management.
		 * The intuition is that program state will override program pipeline state if they are use together,
		 * this state leakage can be painful to debug when mixing program and program pipeline together in a rendering system.
		 * Therefore we can avoid this by deactivating the program state after using.
		 * @return The program active state manager.
		 * Application should call any GL function that uses program state within guard of this state manager.
		*/
		[[nodiscard]] STPProgramStateManager useManaged() const noexcept;

	};

}
#include "STPProgramManager.inl"
#endif//_STP_PROGRAM_MANAGER_H_