#include <SuperRealism+/Scene/Component/STPBidirectionalScattering.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

using glm::uvec2;
using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::uvec4;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto BSDFShaderFilename =
	SuperTerrainPlus::STPFile::generateFilename(STPRealismInfo::ShaderPath, "/STPBidirectionalScattering", ".frag");

STPBidirectionalScattering::STPBidirectionalScattering(const STPScreen::STPScreenInitialiser& screen_init) : BufferDimension(uvec2(0u)) {
	/* --------------------------------------- build shader ----------------------------------------------- */
	const char* const scattering_shader_file = BSDFShaderFilename.data();
	STPShaderManager::STPShaderSource scattering_source(scattering_shader_file , STPFile::read(scattering_shader_file));

	STPShaderManager scattering_shader(GL_FRAGMENT_SHADER);
	scattering_shader(scattering_source);
	this->BSDFQuad.initScreenRenderer(scattering_shader, screen_init);

	/* -------------------------------------- setup sampler ---------------------------------------------- */
	auto setSampler = [](STPSampler& sampler, const auto& border) -> void {
		sampler.wrap(GL_CLAMP_TO_BORDER);
		sampler.borderColor(border);
		sampler.filter(GL_NEAREST, GL_NEAREST);
	};
	constexpr static vec4 BlackColour = vec4(vec3(0.0f), 1.0f);
	setSampler(this->RawSceneColorCopier.ScreenColorSampler, BlackColour);
	setSampler(this->RawSceneDepthCopier.ScreenColorSampler, BlackColour);
	//make the default normal to be facing the camera
	setSampler(this->NormalSampler, vec4(vec2(0.0f), vec2(1.0f)));
	setSampler(this->MaterialSampler, uvec4(0u));

	/* ---------------------------------------- setup uniform ---------------------------------------------- */
	this->BSDFQuad.OffScreenRenderer.uniform(glProgramUniform1i, "ObjectDepth", 0)
		.uniform(glProgramUniform1i, "ObjectNormal", 1)
		.uniform(glProgramUniform1i, "ObjectMaterial", 2);
}

void STPBidirectionalScattering::setScattering(const STPEnvironment::STPBidirectionalScatteringSetting& scattering_setting) {
	if (!scattering_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Setting for bidirectional scattering fails to be validated");
	}

	this->BSDFQuad.OffScreenRenderer.uniform(glProgramUniform1f, "MaxDistance", scattering_setting.MaxRayDistance)
		.uniform(glProgramUniform1f, "DepthBias", scattering_setting.DepthBias)
		.uniform(glProgramUniform1ui, "StepResolution", scattering_setting.RayResolution)
		.uniform(glProgramUniform1ui, "StepSize", scattering_setting.RayStep);
}

void STPBidirectionalScattering::setCopyBuffer(uvec2 dimension) {
	//use floating point to support HDR colour
	this->RawSceneColorCopier.setScreenBuffer(nullptr, dimension, GL_RGB16F);
	this->RawSceneDepthCopier.setScreenBuffer(nullptr, dimension, GL_R32F);

	//update buffer dimension
	this->BufferDimension = dimension;
	//send new handle
	this->BSDFQuad.OffScreenRenderer.uniform(glProgramUniformHandleui64ARB, "SceneColor", *this->RawSceneColorCopier.ScreenColorHandle)
		.uniform(glProgramUniformHandleui64ARB, "SceneDepth", *this->RawSceneDepthCopier.ScreenColorHandle);
}

void STPBidirectionalScattering::copyScene(const STPTexture& color, const STPTexture& depth) {
	const vec2 dim = static_cast<vec2>(this->BufferDimension);

	//experiment shows copy by draw command yields the best performance
	//copy colour
	this->RawSceneColorCopier.capture();
	glDrawTextureNV(*color, *this->RawSceneColorCopier.ScreenColorSampler, 0.0f, 0.0f, dim.x, dim.y, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	//copy depth
	this->RawSceneDepthCopier.capture();
	glDrawTextureNV(*depth, *this->RawSceneDepthCopier.ScreenColorSampler, 0.0f, 0.0f, dim.x, dim.y, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f);

	//reset framebuffer binding point
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
}

void STPBidirectionalScattering::scatter(const STPTexture& depth, const STPTexture& normal, const STPTexture& material) const {
	depth.bind(0);
	normal.bind(1);
	material.bind(2);
	const STPSampler::STPSamplerUnitStateManager depth_normal_material_sampler_mgr[3] = {
		this->RawSceneDepthCopier.ScreenColorSampler.bindManaged(0),
		this->NormalSampler.bindManaged(1),
		this->MaterialSampler.bindManaged(2)
	};

	this->BSDFQuad.drawScreen();
}