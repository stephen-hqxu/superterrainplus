//TEMPLATE DEFINITION FOR SCENE PIPELINE, DO NOT INCLUDE MANUALLY
#ifdef _STP_SCENE_PIPELINE_H_

//Let the user include this
//#include <glad/glad.h>

template<SuperTerrainPlus::STPRealism::STPScenePipeline::STPRenderComponent R>
inline void SuperTerrainPlus::STPRealism::STPScenePipeline::traverse(const STPSceneWorkflow& workflow) const {
	//a helper function to determine if a specific "provide" flag is set against "check".
	static auto getFlag = [](auto check) constexpr -> bool {
		return (R & check) != 0u;
	};

	//update buffer
	this->updateBuffer();
	//retrieve bit flags
	static constexpr bool hasSun = getFlag(STPScenePipeline::RenderComponentSun),
		hasTerrain = getFlag(STPScenePipeline::RenderComponentTerrain),
		hasPost = getFlag(STPScenePipeline::RenderComponentPostProcess);

	//process rendering components.
	//clear the canvas before drawing the new scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (const auto& light : this->ShadowManager.Light) {
		//clear shadow map
		light->clearLightSpace();
	}
	if constexpr (hasPost) {
		//clear post process buffer
		workflow.PostProcess->clear();
	}

	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	/* ------------------------------------------ shadow pass -------------------------------- */
	{
		//record the original viewport size
		const glm::ivec4 pre_vp = this->getViewport();

		for (const auto& light : this->ShadowManager.Light) {
			light->captureLightSpace();

			const glm::uvec2 sha_vp = light->LightFrustum.Resolution;
			//change the view port to fit the shadow map
			glViewport(0, 0, sha_vp.x, sha_vp.y);

			//for those opaque render components (those can cast shadow), the engine provdes a non-shadow and shadow version
			//The shadow version is usually an inherited version of the non-shadow version.
			//The non-shadow version has a render to depth virtual function with no body, so it does absolutely nothing 
			//if the user decides to not render shadow for this particular target.
			if constexpr (hasTerrain) {
				workflow.Terrain->renderDepth();
			}

		}

		//rollback the previous viewport size
		glViewport(pre_vp.x, pre_vp.y, pre_vp.z, pre_vp.w);
		//stop drawing shadow
		STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	}
	/* --------------------------------------------------------------------------------------- */

	//for the rest of the pipeline, we want to render everything onto a post processing buffer
	if constexpr (hasPost) {
		//render everything onto the post process buffer
		workflow.PostProcess->capture();
	}
	/* ------------------------------------ opaque object rendering ----------------------------- */
	if constexpr (hasTerrain) {
		//ready for rendering
		workflow.Terrain->render();
	}

	/* ------------------------------------- environment rendeing ----------------------------- */
	//there is a early depth test optimisation for sun rendering, so leave it to be drawn at the end of the pipeline
	if constexpr (hasSun) {//sun scatters its light
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);
		(*workflow.Sun)();
	}

	/* -------------------------------------- post processing -------------------------------- */
	if constexpr (hasPost) {
		STPFrameBuffer::unbind(GL_FRAMEBUFFER);
		//back buffer is empty, render post processed buffer
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		(*workflow.PostProcess)();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
	}
}

#endif//_STP_SCENE_PIPELINE_H_