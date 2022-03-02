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

	public:

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
		 * @brief Bind stages of a program object to the current program pipeline.
		 * @param stage Specifies a set of program stages to bind to the program pipeline object. 
		 * @param program Specifies the program object containing the shader executables to use in pipeline. 
		 * @return The pointer to the current pipeline manager for chaining.
		*/
		STPPipelineManager& stage(STPOpenGL::STPbitfield, const STPProgramManager&);

		/**
		 * @brief Instantiate a program pipeline. All previous pipeline instances will be removed.
		 * @return The log from instantiation of the pipeline, if any.
		 * Log generated during pipeline building will be reflected to the log handler.
		 * @see STPLogHandler
		*/
		void finalise();

		/**
		 * @brief Get the underlying program pipeline object.
		 * @return The program pipeline object.
		*/
		STPOpenGL::STPuint operator*() const;

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