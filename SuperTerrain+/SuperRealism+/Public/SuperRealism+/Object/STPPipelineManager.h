#pragma once
#ifndef _STP_PIPELINE_MANAGER_H_
#define _STP_PIPELINE_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>

//Program
#include "STPProgramManager.h"

#include <utility>
#include <initializer_list>

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

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		typedef STPSmartGLuintObject<STPPipelineDeleter> STPSmartPipeline;
		//A program pipeline
		STPSmartPipeline Pipeline;

	public:

		//Selects from a shader program whose stages should be mounted onto the pipeline.
		typedef std::pair<STPOpenGL::STPbitfield, const STPProgramManager*> STPPipelineStage;

		/**
		 * @brief Initialise an empty program pipeline object.
		*/
		STPPipelineManager() = default;

		/**
		 * @brief Initialise a program pipeline object and select shader stages from a range of program.
		 * @param stage_program An array of stage-program pair to be mounted onto the pipeline.
		 * @param count The number of element in the array.
		*/
		STPPipelineManager(const STPPipelineStage*, size_t);

		//An array of pipeline stage information
		//@see STPPipelineManager
		STPPipelineManager(std::initializer_list<const STPPipelineStage>);

		STPPipelineManager(const STPPipelineManager&) = delete;

		STPPipelineManager(STPPipelineManager&&) noexcept = default;

		STPPipelineManager& operator=(const STPPipelineManager&) = delete;

		STPPipelineManager& operator=(STPPipelineManager&&) noexcept = default;

		~STPPipelineManager() = default;

		/**
		 * @brief Get the underlying program pipeline object.
		 * @return The program pipeline object.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		/**
		 * @brief Bind the current program pipeline to the context to make it active.
		*/
		void bind() const noexcept;

		/**
		 * @brief Reset the program pipeline to default state, meaning no pipeline will be active.
		*/
		static void unbind() noexcept;

	};

}
#endif//_STP_PIPELINE_MANAGER_H_