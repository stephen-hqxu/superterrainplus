#include <SuperRealism+/Scene/Light/STPLightShadow.h>

//Error
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPInvalidEnum.h>

//GLAD
#include <glad/glad.h>

using glm::uvec2;
using glm::vec3;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

//clear depth colour to zero
constexpr static vec4 BlackColour = vec4(vec3(0.0f), 1.0f);

STPLightShadow::STPLightShadow(const unsigned int resolution, const STPShadowMapFormat format) :
	ShadowMapFormat(format), ShadowMapResolution(resolution), ShadowMapShouldUpdate(true), ShadowMapUpdateMask(true) {
	STP_ASSERTION_NUMERIC_DOMAIN(this->ShadowMapResolution > 0u, "The resolution of the shadow map should be a positive integer");
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPLightShadow::shadowMapHandle() const {
	return this->ShadowMapHandle.get();
}

void STPLightShadow::setShadowMap(const STPShadowMapFilter shadow_filter, const STPOpenGL::STPint level, const STPOpenGL::STPfloat anisotropy) {
	//determine the texture target to be used based on shadow map format.
	GLenum shadow_target = 0u;
	switch (this->ShadowMapFormat) {
	case STPShadowMapFormat::Scalar: shadow_target = GL_TEXTURE_2D;
		break;
	case STPShadowMapFormat::Array: shadow_target = GL_TEXTURE_2D_ARRAY;
		break;
	case STPShadowMapFormat::Cube: shadow_target = GL_TEXTURE_CUBE_MAP;
		break;
	default:
		throw STP_INVALID_ENUM_CREATE(this->ShadowMapFormat, STPShadowMapFormat);
	}
	//determine texture internal format based on shadow map filter
	//For VSM-related shadow mapping, we should use 2-channel colour format rather than depth format.
	const bool useColorInternal = shadow_filter >= STPShadowMapFilter::VSM;
	const GLenum shadow_internal = useColorInternal ? GL_RG16 : GL_DEPTH_COMPONENT24;

	/* ------------------------------ depth texture setup ------------------------- */
	STPTexture shadow_map(shadow_target);
	//shadow map is a square texture
	const uvec2 dimension = uvec2(this->ShadowMapResolution);
	if (this->ShadowMapFormat == STPShadowMapFormat::Scalar) {
		shadow_map.textureStorage2D(level, shadow_internal, dimension);
	} else {
		shadow_map.textureStorage3D(level, shadow_internal, STPGLVector::STPsizeiVec3(dimension, this->lightSpaceDimension()));
	}

	/* ---------------------------- depth texture parameter ------------------------ */
	//texture filtering settings
	switch (shadow_filter) {
	case STPShadowMapFilter::Nearest:
		shadow_map.filter(GL_NEAREST, GL_NEAREST);
		break;
	case STPShadowMapFilter::VSM:
	case STPShadowMapFilter::ESM:
		shadow_map.anisotropy(anisotropy);
		shadow_map.filter(level > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR, GL_LINEAR);
		break;
	default:
		//all other filter options implies linear filtering.
		shadow_map.filter(GL_LINEAR, GL_LINEAR);
		break;
	}

	//others
	shadow_map.wrap(GL_CLAMP_TO_BORDER);
	shadow_map.borderColor(BlackColour);
	if (!useColorInternal) {
		//enable depth sampler for depth-based shadow maps
		//setup compare function so we can use shadow sampler in the shader
		shadow_map.compareFunction(GL_GREATER);
		shadow_map.compareMode(GL_COMPARE_REF_TO_TEXTURE);
	}

	/* --------------------------- depth texture framebuffer -------------------------- */
	if (useColorInternal) {
		//write to colour instead of depth channel
		this->ShadowMapContainer.attach(GL_COLOR_ATTACHMENT0, shadow_map, 0);
	} else {
		//attach the new depth texture to the framebuffer
		this->ShadowMapContainer.attach(GL_DEPTH_ATTACHMENT, shadow_map, 0);
		//we are rendering shadow and colours are not needed.
		this->ShadowMapContainer.drawBuffer(GL_NONE);
		this->ShadowMapContainer.readBuffer(GL_NONE);
	}

	this->ShadowMapContainer.validate(GL_FRAMEBUFFER);

	/* --------------------------- setup depth handle -------------------------------- */
	this->ShadowMapHandle = STPBindlessTexture::make(shadow_map);
	this->updateShadowMapHandle(this->ShadowMapHandle.get());

	using std::move;
	//store the target as member
	this->ShadowMap.emplace(move(shadow_map));
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPLightShadow::shadowDataAddress() const {
	return this->ShadowDataAddress;
}

void STPLightShadow::captureDepth() const {
	this->ShadowMapContainer.bind(GL_FRAMEBUFFER);
}

void STPLightShadow::clearShadowMapColor() {
	this->ShadowMapContainer.clearColor(0, BlackColour);
}

void STPLightShadow::generateShadowMipmap() {
	this->ShadowMap->generateMipmap();
}