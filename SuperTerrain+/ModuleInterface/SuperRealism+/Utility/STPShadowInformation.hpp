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

}
#endif//_STP_SHADOW_INFORMATION_HPP_