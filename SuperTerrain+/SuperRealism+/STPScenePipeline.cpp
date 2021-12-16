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
	float _padPod;
	mat4 V;
	mat4 P;

};

STPScenePipeline::STPScenePipeline(const STPCamera& camera, const STPSceneWorkflow& scene) : SceneCamera(camera), Workflow(scene) {
	//set up buffer for camera transformation matrix
	this->CameraBuffer.bufferStorage(sizeof(STPPackedCameraBuffer), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	this->CameraBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0u);
	this->MappedCameraBuffer = 
		this->CameraBuffer.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
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

void STPScenePipeline::traverse() {
	//clear the canvas before drawing the new scene
	this->reset();

	const vec3& position = this->SceneCamera.cameraStatus().Position;
	//update camera
	STPPackedCameraBuffer* camBuf = reinterpret_cast<STPPackedCameraBuffer*>(this->MappedCameraBuffer);
	//for simplicity, always update position
	camBuf->Pos = position;

	//for the matrices, only update when necessary
	const STPCamera::STPMatrixResult V = this->SceneCamera.view(), 
		P = this->SceneCamera.projection();
	if (!V.second) {
		//view matrix has changed
		camBuf->V = *V.first;
	}
	if (!P.second) {
		//projection matric has changed
		camBuf->P = *P.first;
	}

	//process rendering components.
	//remember that rendering components are optionally nullptr
	if (this->Workflow.Terrain) {
		//prepare for terrain rendering
		this->Workflow.Terrain->prepare(position);

		glDepthFunc(GL_LESS);
		glEnable(GL_CULL_FACE);
		//ready for rendering
		(*this->Workflow.Terrain)();
	}

	if (this->Workflow.Sun) {
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);
		(*this->Workflow.Sun)();
	}
}