#include <SuperRealism+/Scene/Light/STPCascadedShadowMap.h>
//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>
//Algebra
#include <SuperTerrain+/Utility/Algebra/STPVector4d.h>
#include <SuperTerrain+/Utility/Algebra/STPMatrix4x4d.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

//System
#include <numeric>
#include <limits>
#include <functional>
#include <algorithm>

using glm::vec3;
using glm::dvec3;
using glm::mat4;
using glm::dmat4;
using glm::vec4;
using glm::dvec4;

using std::numeric_limits;

using SuperTerrainPlus::STPMatrix4x4d;
using namespace SuperTerrainPlus::STPRealism;

/**
 * @brief CSM data buffer packed according to alignment rule specified in GL_NV_shader_buffer_load.
 * This struct only contains fixed data, variable length data are appended at the end.
*/
struct STPPackedCSMBufferHeader {
public:

	GLuint64 TexHandle;
	unsigned int LiDim;
	unsigned int _padLiDim;
	GLuint64EXT LiSpacePtr, DivPtr;
	//mat4 LiSpace[LiDim];
	//float Div[LiDim - 1];

};

/**
 * @brief Convert the distance in the light frustum from relative distance regarding the far plane to absolute distance.
 * Remember to call this function to refresh the light frustum if `far` of camera changes.
 * @param frustum The frustum containing the relative distance.
 * @return The frustum with absolute distance.
*/
static STPCascadedShadowMap::STPLightFrustum transformFrustum(STPCascadedShadowMap::STPLightFrustum frustum) {
	auto& [div, band, focus, distance] = frustum;
	const double view_far = focus->cameraStatus().Far;

	//transform all relative distance to absolute with respect to the far frustum plane
	std::transform(div.cbegin(), div.cend(), div.begin(), [view_far](const double div_far) { return div_far * view_far; });
	band *= view_far;

	div.shrink_to_fit();
	return frustum;
}

STPCascadedShadowMap::STPCascadedShadowMap(const unsigned int resolution, const STPLightFrustum& light_frustum) :
	STPLightShadow(resolution, STPShadowMapFormat::Array), LightDirection(vec3(0.0f)), FocusEventData { }, LightFrustum(transformFrustum(light_frustum)) {
	const auto& [div, band_radius, focus_camera, distance_mul] = this->LightFrustum;
	if (distance_mul < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}
	if (div.size() == 0u) {
		throw STPException::STPBadNumericRange("There is no shadow level being defined");
	}
	if (band_radius < 0.0f) {
		throw STPException::STPBadNumericRange("Shadow cascade band radius must be non-negative");
	}
	//register a camera callback
	focus_camera->subscribe(this->FocusEventData);

	/* -------------------------------- shadow data buffer allocation ---------------------------- */
	const size_t lightSpaceDim = this->lightSpaceDimension(),
		shadowBufferMat_size = sizeof(mat4) * lightSpaceDim,
		shadowBufferMat_offset = sizeof(STPPackedCSMBufferHeader),
		shadowBufferDiv_offset = shadowBufferMat_offset + shadowBufferMat_size,
		shadowBuffer_size = shadowBufferDiv_offset + sizeof(float) * (lightSpaceDim - 1u);

	//allocate memory shadow data buffer for cascaded shadow map
	this->ShadowData.bufferStorage(shadowBuffer_size, 
		GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	//grab the address of buffer
	this->ShadowDataAddress = this->ShadowData.getAddress();
	this->ShadowData.makeResident(GL_READ_ONLY);

	/* ----------------------------------- initial shadow data fill up --------------------------------------- */
	unsigned char* const shadowData_init = reinterpret_cast<unsigned char*>(this->ShadowData.mapBufferRange(0, shadowBuffer_size, 
		GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
	if (!shadowData_init) {
		throw STPException::STPGLError("Unable to map shadow data buffer to setup initial data for cascaded shadow map");
	}

	//zero fill
	memset(shadowData_init, 0x00, shadowBuffer_size);
	//fixed data header
	STPPackedCSMBufferHeader* const dataHeader = reinterpret_cast<STPPackedCSMBufferHeader*>(shadowData_init);
	dataHeader->LiDim = static_cast<unsigned int>(lightSpaceDim);
	dataHeader->LiSpacePtr = this->ShadowDataAddress + shadowBufferMat_offset;
	dataHeader->DivPtr = this->ShadowDataAddress + shadowBufferDiv_offset;

	//skip light space matrix, send frustum divisor
	float* const shadowData_div = reinterpret_cast<float*>(shadowData_init + shadowBufferDiv_offset);
#pragma warning(push)
#pragma warning(disable: 4244)//precision lost from double -> float
	std::copy(div.cbegin(), div.cend(), shadowData_div);
#pragma warning(pop)

	//flush data, with persistent mapping it is allowed to perform buffer subdata operation
	this->ShadowData.unmapBuffer();
	/* ------------------------------------------------------------------------------------------------------ */

	//assigned the light space matrix pointer
	this->LightSpaceMatrix = reinterpret_cast<mat4*>(this->ShadowData.mapBufferRange(shadowBufferMat_offset, shadowBufferMat_size,
		GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
	if (!this->LightSpaceMatrix) {
		throw STPException::STPGLError("Unable to map the shadow data buffer for cascaded shadow map");
	}
}

using std::move;

STPCascadedShadowMap::STPCascadedShadowMap(STPCascadedShadowMap&& csm) noexcept : 
	STPLightShadow(move(csm)), LightDirection(csm.LightDirection), FocusEventData(csm.FocusEventData), LightFrustum(move(csm.LightFrustum)), 
	LightSpaceMatrix(csm.LightSpaceMatrix) {
	//register the new listener
	this->LightFrustum.Focus->subscribe(this->FocusEventData);
}

STPCascadedShadowMap::~STPCascadedShadowMap() {
	this->LightFrustum.Focus->unsubscribe(this->FocusEventData);
}

mat4 STPCascadedShadowMap::calcLightSpace(const double near, const double far, const STPMatrix4x4d& view) const {
	//min and max of float
	static constexpr double minD = numeric_limits<double>::min(),
		maxD = numeric_limits<double>::max();
	//unit frustum corner, as a lookup table to avoid re-computation.
	const static STPFrustumCorner<STPVector4d> unitCorner = []() -> auto {
		STPFrustumCorner<STPVector4d> frustumCorner = { };

		unsigned int index = 0u;
		for (unsigned int x = 0u; x < 2u; x++) {
			for (unsigned int y = 0u; y < 2u; y++) {
				for (unsigned int z = 0u; z < 2u; z++) {
					frustumCorner[index++] = STPVector4d(
						//the frustum box fits the NDC clip volume
						//note that we are working with zero-to-one depth
						dvec4(
							2.0 * x - 1.0,
							2.0 * y - 1.0,
							z,
							1.0
						)
					);
				}
			}
		}

		return frustumCorner;
	}();

	//calculate the projection matrix for this view frustum
	const STPMatrix4x4d projection = this->LightFrustum.Focus->projection(near, far);
	STPFrustumCorner<STPVector4d> corner;
	std::transform(unitCorner.cbegin(), unitCorner.cend(), corner.begin(), [inv = (projection * view).inverse()](const auto& v) {
		//convert from normalised device coordinate to clip space
		const STPVector4d pt = inv * v;
		return pt / pt.broadcast<STPVector4d::STPElement::W>();
	});

	//each corner is normalised, so sum them up to get the coordinate of centre
	const dvec3 centre =
		static_cast<dvec4>(std::reduce(corner.cbegin(), corner.cend(), STPVector4d(), std::plus<STPVector4d>()))
		/ (1.0 * corner.size());
	alignas(STPMatrix4x4d) const dmat4 lightView_data =
		glm::lookAt(centre + static_cast<dvec3>(this->LightDirection), centre, dvec3(0.0, 1.0, 0.0));
	const STPMatrix4x4d lightView = STPMatrix4x4d(lightView_data);

	//align the light frustum tightly around the current view frustum
	//we need to find the eight corners of the light frustum.
	dvec3 minExt = dvec3(maxD),
		maxExt = dvec3(minD);
	for (const auto& v : corner) {
		//convert the camera view frustum from world space to light view space
		const dvec3 trf = static_cast<dvec3>(static_cast<dvec4>(lightView * v));

		minExt = glm::min(minExt, trf);
		maxExt = glm::max(maxExt, trf);
	}

	const double zDistance = this->LightFrustum.ShadowDistanceMultiplier;
	//Tune the depth (maximum shadow distance)
	double& minZ = minExt.z, 
		&maxZ = maxExt.z;
	if (minZ < 0.0f) {
		minZ *= zDistance;
	} else {
		minZ /= zDistance;
	}

	if (maxZ < 0.0f) {
		maxZ /= zDistance;
	} else {
		maxZ *= zDistance;
	}

	//finally, we are dealing with a directional light so shadow is parallel
	//near, far clipping planes are swapped due to use of reversed depth
	alignas(STPMatrix4x4d) const dmat4 lightProjection_data = glm::orthoRH_ZO(minExt.x, maxExt.x, minExt.y, maxExt.y, maxZ, minZ);
	const STPMatrix4x4d lightProjection = STPMatrix4x4d(lightProjection_data);

	return static_cast<mat4>(lightProjection * lightView);
}

void STPCascadedShadowMap::calcAllLightSpace(mat4* const light_space) const {
	const STPCamera& viewer = *this->LightFrustum.Focus;
	const auto& shadow_level = this->LightFrustum.Division;
	//this offset pushes the far plane away and near plane in
	const double level_offset = this->LightFrustum.CascadeBandRadius;

	//The camera class has smart cache to the view matrix.
	const STPMatrix4x4d& camView = viewer.view();
	const STPEnvironment::STPCameraSetting& camSetting = viewer.cameraStatus();
	const double near = camSetting.Near, far = camSetting.Far;

	const size_t lightSpaceCount = this->lightSpaceDimension();
	//calculate the light view matrix for each subfrusta
	for (unsigned int i = 0u; i < lightSpaceCount; i++) {
		//current light space in the mapped buffer
		mat4& current_light = light_space[i];

		if (i == 0u) {
			//the first frustum
			current_light = this->calcLightSpace(near, shadow_level[i] + level_offset, camView);
		} else if (i < shadow_level.size()) {
			//the middle
			current_light =
				this->calcLightSpace(shadow_level[i - 1u] - level_offset, shadow_level[i] + level_offset, camView);
		} else {
			//the last one
			current_light = this->calcLightSpace(shadow_level[i - 1u] - level_offset, far, camView);
		}
	}
}

inline void STPCascadedShadowMap::requireShadowMapUpdate() {
	if (!this->ShadowMapUpdateMask) {
		//don't modify the update status if the mask if false, i.e., don't trigger automatic update
		return;
	}
	//don't do this:
	//* should_update = true && mask;
	//because it will overwrite the existing flag, if the flag is set manually by user via force update.
	this->ShadowMapShouldUpdate = true;
}

void STPCascadedShadowMap::updateShadowMapHandle(const STPOpenGL::STPuint64 handle) {
	//send the new texture handle to the buffer
	this->ShadowData.bufferSubData(&handle, sizeof(GLuint64), offsetof(STPPackedCSMBufferHeader, TexHandle));
}

void STPCascadedShadowMap::setDirection(const vec3& dir) {
	this->LightDirection = dir;
	this->requireShadowMapUpdate();
}

const vec3& STPCascadedShadowMap::getDirection() const {
	return this->LightDirection;
}

bool STPCascadedShadowMap::updateLightSpace() {
	if (this->FocusEventData.any()) {
		//camera update received
		this->requireShadowMapUpdate();

		this->FocusEventData.unset();
	}

	if (this->ShadowMapShouldUpdate) {
		//need to also update light space matrix if shadow has been turned on for this light
		this->calcAllLightSpace(this->LightSpaceMatrix);

		this->ShadowMapShouldUpdate = false;
		return true;
	}
	return false;
}

inline size_t STPCascadedShadowMap::lightSpaceDimension() const {
	return this->LightFrustum.Division.size() + 1u;
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPCascadedShadowMap::lightSpaceMatrixAddress() const {
	return this->ShadowDataAddress + sizeof(STPPackedCSMBufferHeader);
}