#pragma once
//Include guard
#ifndef _STP_TEXTURE_STORAGE_INCLUDE_
#define _STP_TEXTURE_STORAGE_INCLUDE_
//stb_image Image Loader
#ifndef STBI_INCLUDE_STB_IMAGE_H//this is the include guard of stb_image.h
#include "stb_image.h"//We only need the header for stb_image, definition has been set in the main renderer
#endif

namespace STPDemo {

	/**
	 * @brief A texture storage that contains texture width, height and data
	*/
	struct STPTextureStorage {
	public:

		//The properties of the texture
		const int Width, Height, Channel;
		//the data
		unsigned char* const Texture;

		/**
		 * @brief Putting texture inside one storage class
		 * @param width The width of the texture
		 * @param height The height of the texture
		 * @param channel How many channels this texture has
		 * @param texture The image data
		*/
		STPTextureStorage(const int, const int, const int, unsigned char* const);

		~STPTextureStorage();

		/**
		 * @brief Loadup a texture from the file system
		 * @param filename The filename of the image
		 * @param desiredChannel How many channels required
		 * @return The pointer to the new texture storage, class object needs to be freed after use
		*/
		static STPTextureStorage* loadTexture(const char*, int);

	};
}
#endif//_STP_TEXTURE_STORAGE_INCLUDE_