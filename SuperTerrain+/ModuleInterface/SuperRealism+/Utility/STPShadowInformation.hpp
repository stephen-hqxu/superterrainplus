#ifndef _STP_SHADOW_INFORMATION_HPP_
#define _STP_SHADOW_INFORMATION_HPP_

#include <string>
#include <list>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShadowInformation contains data shared among all renderers for performing shadow mapping.
	 * It is a dictionary of configurations for compiling GLSL shadow map header, with key as setting name and value as setting value.
	*/
	typedef std::list<std::pair<std::string, unsigned int>> STPShadowInformation;
	/**
	 * @brief Defines an optional STPShadowInformation which can be used as an input type to a renderer.
	 * As a general semantics, this setting should not be set (or it may incur some overhead) when shadow is not turned on for a particular renderer, 
	 * and is mandatory when shadow is used.
	*/
	typedef const STPShadowInformation* STPShadowInformation_opt;

}
#endif//_STP_SHADOW_INFORMATION_HPP_