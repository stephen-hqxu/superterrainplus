#pragma once
#ifndef _STP_SHADER_INCLUDE_MANAGER_H_
#define _STP_SHADER_INCLUDE_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>

//String
#include <string>

#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPShaderIncludeManager is a simple helper for shading language include.
	 * It helps building the include file system fopr GLSL shader.
	 * This functionality requires support for ARB_shading_language_include rendering GPU context.
	 * Note that this manager does not handle any GL error, user should either register a debug callback or 
	 * get GL error manually.
	 * @see https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shading_language_include.txt
	*/
	namespace STPShaderIncludeManager {

		/**
		 * @brief STPManagedSourceDeleter is a smart deleter for registered shader include source code from the GL file system.
		*/
		struct STP_REALISM_API STPManagedSourceDeleter {
		private:

			//The length of the pathname.
			const size_t PathLength;

		public:

			/**
			 * @brief Init a source deleter.
			 * @param length The length of the path name in byte.
			*/
			STPManagedSourceDeleter(size_t);

			~STPManagedSourceDeleter() = default;

			void operator()(const char*) const;

		};
		//An auto-deleter for GL named string, it removed managed source from GL automatically without the need to call removeSource().
		typedef std::unique_ptr<const char, STPManagedSourceDeleter> STPManagedSource;

		/**
		 * @brief Check if the current context has shader include extension.
		 * @return True if GPU has support, otherwise false.
		*/
		STP_REALISM_API int support();

		/**
		 * @brief Add a named string, i.e., shader include source code, to the current GL context.
		 * @param path The include path to the virtual GL file system. The pathname needs to be a valid pathname according to GL specification.
		 * @param source The source code (named string) for this file.
		 * @return True if source has been added. If path has already been associated with another source, nothing is done and false is returned.
		*/
		STP_REALISM_API bool addSource(const std::string&, const std::string&);

		/**
		 * @brief Remove a named string from the current GL context.
		 * @param path The include path to the virtual GL file system where the named string will be removed.
		 * @return True if the path exists and has been removed.
		*/
		STP_REALISM_API bool removeSource(const std::string&);

		/**
		 * @brief Check if a path has been added to the GL context with a source previously.
		 * @param path The path name to be checked.
		 * @return True if the path is associating a source, false otherwise.
		*/
		STP_REALISM_API bool exist(const std::string&);

		/**
		 * @brief Register a pathname with a smart deleter.
		 * Include source is removed from the virtual filesystem automatically when the managed source is destroied.
		 * If the source is removed, all existing smart deleters are invalidated automatically and destruction causes no effect.
		 * If then the source is added back with the same pathname, all invalidated deleters will be come valid again.
		 * @param path The pathname to be registered.
		 * If path is previously been registered, exception is thrown.
		 * @return The smart source manager.
		*/
		STP_REALISM_API STPManagedSource registerRemover(const std::string&);

	}

}
#endif//_STP_SHADER_INCLUDE_MANAGER_H_