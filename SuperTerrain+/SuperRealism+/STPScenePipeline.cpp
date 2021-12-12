#include <SuperRealism+/STPScenePipeline.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

using glm::vec3;
using glm::vec4;
using glm::mat4;

using namespace SuperTerrainPlus::STPRealism;

/**
 * @brief Change a boolean status only when the current status is different from the target.
 * @param type The type of this status to be updated to GL context.
 * @param target_status The new status to change to.
*/
static void changeBinStatus(GLenum type, bool target_status) {
	//check the current status from the context
	const bool current_status = glIsEnabled(type);

	if (current_status != target_status) {
		//status is different, update
		if (target_status) {
			glEnable(type);
		}
		else {
			glDisable(type);
		}
	}
	//nothing needs to be done if status is the same as current.
}

/**
 * @brief Change a value status when the current status is different.
 * @tparam V The type of the status value.
 * @tparam Que The status function to get the current status.
 * @tparam Upd The functio to update the new status.
 * @param status_query The function to query the current status.
 * @param update_function The function to update the status.
 * @param status_type The GL enum type of the status being requested.
 * @param target_status The new status to change to.
 * @param gl_func The GL function to update the status.
*/
template<typename V, class Que, class Upd>
static void changeValueStatus(Que&& status_query, Upd&& update_function, GLenum status_type, V target_status) {
	using std::forward;

	V current_status;
	//check the current status
	forward<Que>(status_query)(status_type, &current_status);

	//update if not the same
	if (current_status != target_status) {
		forward<Upd>(update_function)(target_status);
	}
}

STPScenePipeline::STPScenePipeline(const STPCamera& camera, const STPSceneWorkflow& scene) : SceneCamera(camera), Workflow(scene) {
	constexpr static size_t cameraBufferSize = sizeof(mat4) * 2u + sizeof(vec3);

	//set up buffer for camera transformation matrix
	this->CameraBuffer.bufferStorage(cameraBufferSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->CameraBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0u);
	this->MappedCameraBuffer = 
		this->CameraBuffer.mapBufferRange(cameraBufferSize, 0, 
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
	if (!this->MappedCameraBuffer) {
		throw STPException::STPGLError("Unable to map camera buffer to shader storage buffer");
	}
	//buffer has been setup
	
	//set up initial GL context states
	changeBinStatus(GL_DEPTH_TEST, true);
	changeValueStatus<GLint>(glGetIntegerv, glDepthFunc, GL_DEPTH_FUNC, GL_LESS);
	changeValueStatus<GLboolean>(glGetBooleanv, glDepthMask, GL_DEPTH_WRITEMASK, GL_TRUE);

	changeBinStatus(GL_STENCIL_TEST, false);

	changeBinStatus(GL_CULL_FACE, true);
	changeValueStatus<GLint>(glGetIntegerv, glCullFace, GL_CULL_FACE_MODE, GL_BACK);
	changeValueStatus<GLint>(glGetIntegerv, glFrontFace, GL_FRONT_FACE, GL_CCW);

	changeBinStatus(GL_BLEND, false);
}

STPScenePipeline::~STPScenePipeline() {
	//release the shader storage buffer
	this->CameraBuffer.unmapBuffer();
}

inline void STPScenePipeline::reset() const {
	//clear the canvas
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void STPScenePipeline::setClearColor(vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
}