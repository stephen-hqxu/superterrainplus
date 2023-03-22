#pragma once
#ifndef _STP_MATERIAL_LIBRARY_H_
#define _STP_MATERIAL_LIBRARY_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPBuffer.h"

#include <memory>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPMaterialLibrary contains all registered material data for objects in the scene.
	 * The material library can be linked with the rendering pipeline
	 * so shader can access material property using material ID assigned to each object.
	 * Access to material property with an invalid ID, however, is a undefined behaviour.
	*/
	class STP_REALISM_API STPMaterialLibrary {
	public:

#include "../Shader/Common/STPMaterialRegistry.glsl"

		//An unique identifier to a material registered in the material library
		typedef unsigned char STPMaterialID;
		//An integer to identify the number of material
		typedef STPMaterialID STPMaterialCount;

	private:

		//material properties stored on host as a synchronisation backup
		std::unique_ptr<STPMaterialProperty[]> MaterialMemory;
		STPMaterialCount MaterialCount;
		//material stored on device
		STPBuffer MaterialBuffer;

	public:

		//The allocation size for the material buffer
		const STPMaterialCount MaxMaterialAllowance;

		//The default initialised material property.
		constexpr static STPMaterialProperty DefaultMaterial = {
			1.0f,
			1.0f,
			1.0f
		};

		/**
		 * @brief Initialise a new material library.
		 * The material library by default has the first element filled in with default material.
		 * @param count Specifies the maximum number of material allows to be added to the library.
		 * The number of material may not exceed the numeric limit allowed by material ID.
		 * It is allowed to have zero count, to indicate no user-specified material.
		*/
		STPMaterialLibrary(STPMaterialCount);

		STPMaterialLibrary(const STPMaterialLibrary&) = delete;

		STPMaterialLibrary(STPMaterialLibrary&&) = default;

		STPMaterialLibrary& operator=(const STPMaterialLibrary&) = delete;

		STPMaterialLibrary& operator=(STPMaterialLibrary&&) = default;

		~STPMaterialLibrary() = default;

		/**
		 * @brief Add a new material into the material library.
		 * This operation will fail if the material library has no more space for a new material.
		 * @param mat_prop The pointer to the material property.
		 * @return The material ID to uniquely identify the material in this library.
		 * The material ID will always be greater than zero; ID of zero is reserved as default material.
		*/
		[[nodiscard]] STPMaterialID add(const STPMaterialProperty&);

		/**
		 * @brief Get the current number of registered material.
		 * @return The current number of material.
		*/
		STPMaterialCount size() const noexcept;

		/**
		 * @brief Get the material registered in the library.
		 * @param id The material ID for which the property to be returned.
		 * @return The pointer to the material property.
		 * Index out of bound will result in undefined behaviour.
		*/
		const STPMaterialProperty& operator[](STPMaterialID) const noexcept;

		/**
		 * @brief Get the underlying buffer for the material library.
		 * @return The pointer to the buffer containing material data.
		*/
		const STPBuffer& operator*() const noexcept;

	};

}
#endif//_STP_MATERIAL_LIBRARY_H_