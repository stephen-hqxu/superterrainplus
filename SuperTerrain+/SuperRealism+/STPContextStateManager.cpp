#include <SuperRealism+/Utility/STPContextStateManager.h>

//GLAD
#include <glad/glad.h>

#include <type_traits>

using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

STPContextStateManager::STPContextState::STPContextState() {
	//Load default context states, they are defined by GL specification as default values
	
	//Depth buffer
	this->DepthBuffer.Switch = false;
	this->DepthBuffer.DepthFunc = GL_LESS;
	this->DepthBuffer.DepthMask = GL_TRUE;
	//Cull face
	this->CullFace.Switch = false;
	this->CullFace.FrontFace = GL_CCW;
	this->CullFace.CullFace = GL_BACK;
	//Blend
	this->Blend.Switch = false;
	this->Blend.FuncSource = GL_ONE;
	this->Blend.FuncDestination = GL_ZERO;
	this->Blend.Equation = GL_FUNC_ADD;
	this->Blend.BlendColor = vec4(0.0f);
}

void STPContextStateManager::STPCurrentState::changeBinStatus(bool& current_status, bool target_status, STPOpenGL::STPenum type) {
	if (current_status != target_status) {
		//depth test status is different, update
		if (target_status) {
			glEnable(type);
		}
		else {
			glDisable(type);
		}
		//update current status record
		current_status = target_status;
	}
	//nothing needs to be done if status is the same as current.
}

template<typename V, typename Fun>
void STPContextStateManager::STPCurrentState::changeValueStatus(V& current_status, V target_status, Fun&& type) {
	if (current_status != target_status) {
		std::forward<Fun>(type)(target_status);

		current_status = target_status;
	}
}

STPContextStateManager::STPCurrentState& STPContextStateManager::STPCurrentState::instance() {
	//Because GL only allows one state throughout a context, so we make the current state a singleton.
	static STPCurrentState Instance;
	
	return Instance;
}

const STPContextStateManager::STPContextState& STPContextStateManager::STPCurrentState::currentState() const {
	return *dynamic_cast<const STPContextState*>(this);
}

void STPContextStateManager::STPCurrentState::depthTest(bool status) {
	STPCurrentState::changeBinStatus(this->DepthBuffer.Switch, status, GL_DEPTH_TEST);
}

void STPContextStateManager::STPCurrentState::depthFunc(STPOpenGL::STPenum func) {
	STPCurrentState::changeValueStatus(this->DepthBuffer.DepthFunc, func, glDepthFunc);
}

void STPContextStateManager::STPCurrentState::depthMask(STPOpenGL::STPboolean mask) {
	STPCurrentState::changeValueStatus(this->DepthBuffer.DepthMask, mask, glDepthMask);
}

void STPContextStateManager::STPCurrentState::cullFace(bool status) {
	STPCurrentState::changeBinStatus(this->CullFace.Switch, status, GL_CULL_FACE);
}

void STPContextStateManager::STPCurrentState::frontFace(STPOpenGL::STPenum mode) {
	STPCurrentState::changeValueStatus(this->CullFace.FrontFace, mode, glFrontFace);
}

void STPContextStateManager::STPCurrentState::faceCulled(STPOpenGL::STPenum mode) {
	STPCurrentState::changeValueStatus(this->CullFace.CullFace, mode, glCullFace);
}

void STPContextStateManager::STPCurrentState::blend(bool status) {
	STPCurrentState::changeBinStatus(this->Blend.Switch, status, GL_BLEND);
}

void STPContextStateManager::STPCurrentState::blendFunc(STPOpenGL::STPenum sfactor, STPOpenGL::STPenum dfactor) {
	if (this->Blend.FuncSource != sfactor && this->Blend.FuncDestination != dfactor) {
		glBlendFunc(sfactor, dfactor);
		//update
		this->Blend.FuncSource = sfactor;
		this->Blend.FuncDestination = dfactor;
	}
}

void STPContextStateManager::STPCurrentState::blendEquation(STPOpenGL::STPenum mode) {
	STPCurrentState::changeValueStatus(this->Blend.Equation, mode, glBlendEquation);
}

void STPContextStateManager::STPCurrentState::blendColor(vec4 color) {
	if (this->Blend.BlendColor != color) {
		glBlendColor(color.r, color.g, color.b, color.a);

		this->Blend.BlendColor = color;
	}
}