#pragma once
#ifndef _STP_PIPELINE_MANAGER_H_
#define _STP_PIPELINE_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//Compatibility
#include <SuperTerrain+/STPOpenGL.h>

//Program
#include "STPProgramManager.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPPipelineManager is a smart manager to GL program pipeline objects.
	 * It takes different shader stages from a few separable programs and recombine them into a single execution pipeline.
	*/
	class STP_REALISM_API STPPipelineManager {
	private:

		/**
		 * @brief STPPipelineDeleter calls glDeleteProgramPipeline to remove a pipeline.
		*/
		struct STP_REALISM_API STPPipelineDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPPipelineDeleter>> STPSmartPipeline;
		//A program pipeline
		STPSmartPipeline Pipeline;

		//Pipeline linking log
		std::string Log;

	public:

		//STPShaderSelection records the shader stages being picked from various programs
		typedef std::list<std::pair<STPOpenGL::STPbitfield, const STPProgramManager*>> STPShaderSelection;

		/**
		 * @brief Initialise a new program pipeline object.
		*/
		STPPipelineManager();

		STPPipelineManager(const STPPipelineManager&) = delete;

		STPPipelineManager(STPPipelineManager&&) noexcept = default;

		STPPipelineManager& operator=(const STPPipelineManager&) = delete;

		STPPipelineManager& operator=(STPPipelineManager&&) noexcept = default;

		~STPPipelineManager() = default;

		/**
		 * @brief Instantiate a program pipeline. All previous pipeline instances will be removed.
		 * @param stages An array of shader bits selected from any given programs.
		 * @return The log from instantiation of the pipeline, if any.
		 * If instantiation fails, exception is thrown with error log.
		*/
		const std::string& operator()(const STPShaderSelection&);

		/**
		 * @brief Get the underlying program pipeline object.
		 * @return The program pipeline object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Retrieve the last pipeling log during initialisation.
		 * @return Pipeline log.
		*/
		const std::string& lastLog() const;

		/**
		 * @brief Bind the current program pipeline to the context to make it active.
		*/
		void bind() const;

		/**
		 * @brief Reset the program pipeline to default state, meaning no pipeline will be active.
		*/
		static void unbind();

	};

}
#endif//_STP_PIPELINE_MANAGER_H_