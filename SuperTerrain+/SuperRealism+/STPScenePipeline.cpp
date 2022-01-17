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
 * @brief Packed struct for mapped camera buffer following OpenGL std430 alignment rule.
*/
struct STPPackedCameraBuffer {
public:

	vec3 Pos;
	float _padPos;
	mat4 V;
	mat4 P;

};

STPScenePipeline::STPScenePipeline(const STPCamera& camera) : SceneCamera(camera), updatePosition(true), updateView(true), updateProjection(true) {
	//set up buffer for camera transformation matrix
	this->CameraBuffer.bufferStorage(sizeof(STPPackedCameraBuffer), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->CameraBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0u);
	this->MappedCameraBuffer = 
		this->CameraBuffer.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	if (!this->MappedCameraBuffer) {
		throw STPException::STPGLError("Unable to map camera buffer to shader storage buffer");
	}
	//buffer has been setup, clear the buffer before use
	memset(this->MappedCameraBuffer, 0x00u, sizeof(STPPackedCameraBuffer));
	
	//set up initial GL context states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	//tessellation settings
	//barycentric coordinate system
	glPatchParameteri(GL_PATCH_VERTICES, 3);

	//register camera callback
	this->SceneCamera.registerListener(dynamic_cast<STPCamera::STPStatusChangeCallback*>(this));
}

STPScenePipeline::~STPScenePipeline() {
	//release the shader storage buffer
	this->CameraBuffer.unmapBuffer();
}

void STPScenePipeline::updateBuffer() const {
	STPPackedCameraBuffer* camBuf = reinterpret_cast<STPPackedCameraBuffer*>(this->MappedCameraBuffer);

	//only update buffer when necessary
	if (this->updatePosition || this->updateView) {
		//position has changed
		if (this->updatePosition) {
			camBuf->Pos = this->SceneCamera.cameraStatus().Position;
			this->CameraBuffer.flushMappedBufferRange(0, sizeof(vec3));

			this->updatePosition = false;
		}

		//view matrix has changed
		camBuf->V = this->SceneCamera.view();
		this->CameraBuffer.flushMappedBufferRange(sizeof(vec4), sizeof(mat4));

		this->updateView = false;
	}
	if (this->updateProjection) {
		constexpr static size_t offset_P = sizeof(vec4) + sizeof(mat4);
		//projection matrix has changed
		camBuf->P = this->SceneCamera.projection();
		this->CameraBuffer.flushMappedBufferRange(offset_P, sizeof(mat4));

		this->updateProjection = false;
	}
}

//by using separate flags instead of just flushing the buffer,
//we can avoid flushing frequently if camera is updated multiple times before next frame.

void STPScenePipeline::onMove(const STPCamera&) {
	this->updatePosition = true;
	this->updateView = true;
}

void STPScenePipeline::onRotate(const STPCamera&) {
	this->updateView = true;
}

void STPScenePipeline::onReshape(const STPCamera&) {
	this->updateProjection = true;
}

void STPScenePipeline::setClearColor(vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
}