#ifndef _STP_DEFINE_H_
#define _STP_DEFINE_H_

#ifdef SUPERTERRAINPLUS_USE_STATIC
//No export/import is needed on static library
#define SUPERTERRAINPLUS_API
#else
#if defined (_MSC_VER)
//MSVC compiler
#ifdef SUPERTERRAINPLUS_EXPORTS
#define SUPERTERRAINPLUS_API __declspec(dllexport)
#else
#define SUPERTERRAINPLUS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
//GCC
#ifdef SUPERTERRAINPLUS_EXPORTS
#define SUPERTERRAINPLUS_API __attribute__((visibility("default")))
#else
#define SUPERTERRAINPLUS_API
#endif
#else
//Do nothing for compiler that exports automatically
#define SUPERTERRAINPLUS_API
#endif
#endif//SUPERTERRAINPLUS_USE_STATIC

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief Denotes the SuperAlgorithm+ static library output path
	*/
	constexpr static char SuperAlgorithmPlus_Library = "@SuperAlgorithm+Lib@";
	/**
	 * @brief Denotes the SuperAlgorithm+ include path
	*/
	constexpr static char SuperAlgorithmPlus_Include = "@SuperAlgorithm+Include@";

}
#endif//_STP_DEFINE_H_