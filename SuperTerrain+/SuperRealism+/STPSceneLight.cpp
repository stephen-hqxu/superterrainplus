#include <SuperRealism+/Scene/STPSceneLight.h>

using std::move;

using namespace SuperTerrainPlus::STPRealism;

STPLightShadow& STPSceneLight::STPGlobalLight<true>::getLightShadow() {
	return const_cast<STPLightShadow&>(const_cast<const STPGlobalLight<true>*>(this)->getLightShadow());
}

STPSceneLight::STPEnvironmentLight<true>::STPEnvironmentLight(STPEnvironmentLightShadow&& env_shadow) : Shadow(move(env_shadow)) {

}

const STPLightShadow& STPSceneLight::STPEnvironmentLight<true>::getLightShadow() const {
	return *this->Shadow;
}