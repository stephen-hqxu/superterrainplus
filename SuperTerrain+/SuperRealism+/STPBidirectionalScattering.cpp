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

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto BSDFShaderFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPBidirectionalScattering", ".frag");

STPBidirectionalScattering::STPBidirectionalScattering(const STPScreenInitialiser& screen_init) : SceneColor(GL_TEXTURE_2D), SceneDepth(GL_TEXTURE_2D), 
	BufferDimension(uvec2(0u)) {
	/* --------------------------------------- build shader ----------------------------------------------- */
	const char* const scattering_shader_file = BSDFShaderFilename.data();
	STPShaderManager::STPShaderSource scattering_source(scattering_shader_file , *STPFile(scattering_shader_file));

	STPShaderManager scattering_shader(GL_FRAGMENT_SHADER);
	scattering_shader(scattering_source);
	this->initScreenRenderer(scattering_shader, screen_init);

	/* -------------------------------------- setup sampler ---------------------------------------------- */
	auto setSampler = [](STPSampler& sampler, const auto& border) -> void {
		sampler.wrap(GL_CLAMP_TO_BORDER);
		sampler.borderColor(border);
		sampler.filter(GL_NEAREST, GL_NEAREST);
	};
	setSampler(this->ColorSampler, vec4(vec3(0.0f), 1.0f));
	setSampler(this->DepthSampler, vec4(1.0f));
	//make the default normal to be facing the camera
	setSampler(this->NormalSampler, vec4(vec2(0.0f), vec2(1.0f)));
	setSampler(this->MaterialSampler, uvec4(0u));

	/* ---------------------------------------- setup uniform ---------------------------------------------- */
	this->OffScreenRenderer.uniform(glProgramUniform1i, "ObjectDepth", 0)
		.uniform(glProgramUniform1i, "ObjectNormal", 1)
		.uniform(glProgramUniform1i, "ObjectMaterial", 2);
}

void STPBidirectionalScattering::setScattering(const STPEnvironment::STPBidirectionalScatteringSetting& scattering_setting) {
	if (!scattering_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Setting for bidirectional scattering fails to be validated");
	}

	this->OffScreenRenderer.uniform(glProgramUniform1f, "MaxDistance", scattering_setting.MaxRayDistance)
		.uniform(glProgramUniform1f, "DepthBias", scattering_setting.DepthBias)
		.uniform(glProgramUniform1ui, "StepResolution", scattering_setting.RayResolution)
		.uniform(glProgramUniform1ui, "StepSize", scattering_setting.RayStep);
}

void STPBidirectionalScattering::setCopyBuffer(uvec2 dimension) {
	if (dimension.x == 0u || dimension.y == 0u) {
		throw STPException::STPBadNumericRange("Both components of the copy buffer resolution must be positive");
	}
	this->BufferDimension = dimension;

	//allocate memory
	const uvec3 dim = uvec3(this->BufferDimension, 1u);
	STPTexture copy_color(GL_TEXTURE_2D), copy_depth(GL_TEXTURE_2D);
	//use floating point to support HDR colour
	copy_color.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16F, dim);
	copy_depth.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R32F, dim);

	//set framebuffer
	this->RawSceneColorCopier.attach(GL_COLOR_ATTACHMENT0, copy_color, 0);
	this->RawSceneDepthCopier.attach(GL_COLOR_ATTACHMENT0, copy_depth, 0);
	if (this->RawSceneColorCopier.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE || this->RawSceneDepthCopier.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Copy framebuffer fails to validate for bidirectional scattering");
	}

	//create new handle
	STPBindlessTexture copy_color_handle(copy_color, this->ColorSampler), copy_depth_handle(copy_depth, this->DepthSampler);
	//send new handle
	this->OffScreenRenderer.uniform(glProgramUniformHandleui64ARB, "SceneColor", *copy_color_handle)
		.uniform(glProgramUniformHandleui64ARB, "SceneDepth", *copy_depth_handle);

	using std::move;
	//replace member with new memory
	this->SceneColor = move(copy_color);
	this->SceneDepth = move(copy_depth);

	this->SceneColorHandle.emplace(move(copy_color_handle));
	this->SceneDepthHandle.emplace(move(copy_depth_handle));
}

void STPBidirectionalScattering::copyScene(const STPTexture& color, const STPTexture& depth) {
	const vec2 dim = static_cast<vec2>(this->BufferDimension);

	//experiment shows copy by draw command yields the best performance
	//copy colour
	this->RawSceneColorCopier.bind(GL_FRAMEBUFFER);
	glDrawTextureNV(*color, *this->ColorSampler, 0.0f, 0.0f, dim.x, dim.y, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	//copy depth
	this->RawSceneDepthCopier.bind(GL_FRAMEBUFFER);
	glDrawTextureNV(*depth, *this->DepthSampler, 0.0f, 0.0f, dim.x, dim.y, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f);

	//reset framebuffer binding point
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
}

void STPBidirectionalScattering::scatter(const STPTexture& depth, const STPTexture& normal, const STPTexture& material) const {
	depth.bind(0);
	normal.bind(1);
	material.bind(2);

	this->DepthSampler.bind(0);
	this->NormalSampler.bind(1);
	this->MaterialSampler.bind(2);

	this->ScreenVertex->bind();
	this->OffScreenRenderer.use();

	this->drawScreen();

	//clear up
	STPSampler::unbind(0);
	STPSampler::unbind(1);
	STPSampler::unbind(2);
	STPProgramManager::unuse();
}