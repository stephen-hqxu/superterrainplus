#pragma once
//Include guard
#ifndef _STP_TEXTURE_STORAGE_INCLUDE_
#define _STP_TEXTURE_STORAGE_INCLUDE_

//System
#include <string>
#include <memory>

//Image Loader
#include <stb_image.h>

//GLM
#include <glm/vec3.hpp>

namespace STPDemo {

	/**
	 * @brief A texture storage that contains texture width, height and data
	*/
	class STPTextureStorage {
	private:

		/**
		 * @brief STPTextureFreer is a deleter for texture image data
		*/
		struct STPTextureFreer {
		public:

			void operator()(stbi_uc*) const;
			
		};

		//Smartly managed texture memory
		typedef std::unique_ptr<stbi_uc, STPTextureFreer> STPTextureMemmory;

		//The properties of the texture
		//Width, Height, Channel
		glm::ivec3 Property;
		//the data
		STPTextureMemmory Texture;

	public:

		/**
		 * @brief Init an empty texture storage.
		*/
		STPTextureStorage() = default;

		/**
		 * @brief Load image of texture from local file system and manage by the current texture storage object.
		 * @param filename The filename to the texture.
		 * @param comp Require components from the source texture.
		*/
		STPTextureStorage(const std::string&, int);

		STPTextureStorage(const STPTextureStorage&) = delete;

		STPTextureStorage(STPTextureStorage&&) noexcept = default;

		STPTextureStorage& operator=(const STPTextureStorage&) = delete;

		STPTextureStorage& operator=(STPTextureStorage&&) noexcept = default;

		~STPTextureStorage() = default;

		/**
		 * @brief Check if the texture container is empty.
		 * @return True if there is no texture being stored in the texture storage.
		*/
		bool empty() const noexcept;

		/**
		 * @brief Get the property of the stored texture.
		 * @return A vector of properties. Being width, height and number of channel for each component.
		*/
		const glm::ivec3& property() const noexcept;

		/**
		 * @brief Get the underlying texture data.
		 * @return The pointer to the texture stored.
		*/
		const stbi_uc* texture() const noexcept;

	};
}
#endif//_STP_TEXTURE_STORAGE_INCLUDE_