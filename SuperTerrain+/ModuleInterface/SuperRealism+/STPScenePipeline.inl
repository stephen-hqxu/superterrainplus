//TEMPLATE DEFINITION FOR SCENE PIPELINE, DO NOT INCLUDE MANUALLY
#ifdef _STP_SCENE_PIPELINE_H_

//Let the user include this
//#include <glad/glad.h>

template<SuperTerrainPlus::STPRealism::STPScenePipeline::STPRenderComponent R, SuperTerrainPlus::STPRealism::STPScenePipeline::STPShadowComponent S>
inline void SuperTerrainPlus::STPRealism::STPScenePipeline::traverse(const STPSceneWorkflow& workflow) const {
	//a helper function to determine if a specific "provide" flag is set against "check".
	static auto getFlag = [](auto flag, auto check) constexpr -> bool {
		return (flag & check) != 0u;
	};

	//update scene buffer
	this->updateBuffer();
	//retrieve bit flags
	static constexpr bool hasSun = getFlag(R, STPScenePipeline::RenderComponentSun),
		hasTerrain = getFlag(R, STPScenePipeline::RenderComponentTerrain),
		hasPost = getFlag(R, STPScenePipeline::RenderComponentPostProcess);
	static constexpr bool shadowTerrain = getFlag(S, STPScenePipeline::ShadowComponentTerrain);

	//process rendering components.
	//clear the canvas before drawing the new scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if constexpr (hasPost) {
		//also clear post process buffer
		workflow.PostProcess->clear();
		//render everything onto the post process buffer
		workflow.PostProcess->capture();
	}

	if constexpr (hasTerrain) {
		glDepthFunc(GL_LESS);
		glEnable(GL_CULL_FACE);
		//ready for rendering
		(*workflow.Terrain)();
	}

	//there is a early depth test optimisation for sun rendering, so leave it to be drawn at the end of the pipeline
	if constexpr (hasSun) {
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);
		(*workflow.Sun)();
	}

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