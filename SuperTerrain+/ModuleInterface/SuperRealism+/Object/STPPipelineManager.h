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

		//Pipeline linking log
		std::unique_ptr<char[]> Log;

	public:

		//STPShaderSelection records the shader stages being picked from various programs
		typedef std::list<std::pair<STPOpenGL::STPbitfield, const STPProgramManager*>> STPShaderSelection;

		//Pipeline object managed.
		const STPOpenGL::STPuint Pipeline;

		/**
		 * @brief Initialise a new program pipeline object.
		 * @param stages An array of shader select.
		 * It determines which shader bits to be used from each program.
		*/
		STPPipelineManager(const STPShaderSelection&);

		STPPipelineManager(const STPPipelineManager&) = delete;

		STPPipelineManager(STPPipelineManager&&) = delete;

		STPPipelineManager& operator=(const STPPipelineManager&) = delete;

		STPPipelineManager& operator=(STPPipelineManager&&) = delete;

		~STPPipelineManager();

		/**
		 * @brief Retrieve any pipeline log during initialisation.
		 * @return Pipeline log.
		*/
		std::optional<std::string_view> getLog() const;

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