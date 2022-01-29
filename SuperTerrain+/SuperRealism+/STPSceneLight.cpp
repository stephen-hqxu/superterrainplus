#include <SuperRealism+/Scene/STPSceneLight.h>

using namespace SuperTerrainPlus::STPRealism;

STPSceneLight::STPEnvironmentLight<true>::STPEnvironmentLight(const STPCascadedShadowMap::STPLightFrustum& frustum) : EnvironmentLightShadow(frustum) {

}

const STPLightShadow& STPSceneLight::STPEnvironmentLight<true>::getLightShadow() const {
	return this->getEnvironmentLightShadow();
}

inline const STPCascadedShadowMap& STPSceneLight::STPEnvironmentLight<true>::getEnvironmentLightShadow() const {
	return this->EnvironmentLightShadow;
}