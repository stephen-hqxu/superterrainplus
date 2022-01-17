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
	if constexpr (hasSun) {
		//clear shadow map
		workflow.Sun->clearLightSpace();
	}
	if constexpr (hasPost) {
		//clear post process buffer
		workflow.PostProcess->clear();
	}

	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	/* ------------------------------------------ shadow pass -------------------------------- */
	if constexpr (hasSun) {//sun casts shadow
		workflow.Sun->captureLightSpace();
		//for shadow to avoid light bleeding, we usually cull front face (with respect to the light)
		glCullFace(GL_FRONT);

		if constexpr (hasTerrain) {
			//TODO: need to work on this, terrain program needs to be splited, and allow user to disable shadow (if sun is not present).
			//workflow.Terrain->renderDepth();
		}

		glCullFace(GL_BACK);
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
		workflow.Terrain->renderShaded();
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