#pragma once
#ifndef _STP_TEXTURE_H_
#define _STP_TEXTURE_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

#include "STPImageParameter.hpp"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPTexture is a thin wrapper to GL texture objects and smartly handle its lifetime.
	*/
	class STP_REALISM_API STPTexture : public STPImageParameter {
	private:

		STPSmartDeviceObject::STPGLTextureObject Texture;

		//The target this texture object is bound to.
		STPOpenGL::STPenum Target;

	public:

		/**
		 * @brief Init a new and empty STPTexture.
		 * @param target Specifies the effective texture target of each created texture.
		*/
		STPTexture(STPOpenGL::STPenum) noexcept;

		STPTexture(const STPTexture&) = delete;

		STPTexture(STPTexture&&) noexcept = default;

		STPTexture& operator=(const STPTexture&) = delete;

		STPTexture& operator=(STPTexture&&) noexcept = default;

		~STPTexture() = default;

		/**
		 * @brief Get the target this texture object is bound to.
		 * @return The texture target bounded.
		*/
		STPOpenGL::STPenum target() const noexcept;

		/**
		 * @brief Get the underlying texture buffer object.
		 * @return The texture object.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		/**
		 * @brief Bind an existing texture object to the specified texture unit 
		 * @param unit Specifies the texture unit, to which the texture object should be bound to. 
		*/
		void bind(STPOpenGL::STPuint) const noexcept;

		/**
		 * @brief Bind a level of a texture to an image unit.
		 * @param unit Specifies the index of the image unit to which to bind the texture.
		 * @param level Specifies the level of the texture that is to be bound.
		 * @param layered Specifies whether a layered texture binding is to be established.
		 * @param layer If layered is GL_FALSE, specifies the layer of texture to be bound to the image unit. Ignored otherwise.
		 * @param access Specifies a token indicating the type of access that will be performed on the image.
		 * @param format Specifies the format that the elements of the image will be treated as for the purposes of formatted stores.
		*/
		void bindImage(STPOpenGL::STPuint, STPOpenGL::STPint, STPOpenGL::STPboolean, STPOpenGL::STPint, STPOpenGL::STPenum, STPOpenGL::STPenum) const noexcept;

		/**
		 * @brief Unbine a texture from the image unit.
		 * @param unit The texture unit to be unbound.
		*/
		static void unbindImage(STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Unbind a texture unit.
		 * @param unit The unit to be unbound.
		*/
		static void unbind(STPOpenGL::STPuint) noexcept;

		/**
		 * @brief Generate mipmaps for a specified texture object
		*/
		void generateMipmap() noexcept;

		void filter(STPOpenGL::STPint, STPOpenGL::STPint) noexcept override;

		void wrap(STPOpenGL::STPint, STPOpenGL::STPint, STPOpenGL::STPint) noexcept override;

		void wrap(STPOpenGL::STPint) noexcept override;

		void borderColor(STPGLVector::STPfloatVec4) noexcept override;

		void borderColor(STPGLVector::STPintVec4) noexcept override;

		void borderColor(STPGLVector::STPuintVec4) noexcept override;

		void anisotropy(STPOpenGL::STPfloat) noexcept override;

		void compareFunction(STPOpenGL::STPint) noexcept override;

		void compareMode(STPOpenGL::STPint) noexcept override;

		/**
		 * @brief Allocate immutable storage for a texture.
		 * @tparam Dim The dimension of the texture, it reflects the glTextureStorage function to be called.
		 * @param level Specify the number of texture levels.
		 * @param internal Specifies the sized internal format to be used to store texture image data.
		 * @param dimension Specifies the width, height, depth of the texture, in texels. 
		 * For 1D texture, y and z component are ignored.
		 * For 2D or 1D array texture, z component is ignored.
		 * Used component should not be zero.
		*/
		void textureStorage1D(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPOpenGL::STPsizei) noexcept;
		void textureStorage2D(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPGLVector::STPsizeiVec2) noexcept;
		void textureStorage3D(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPGLVector::STPsizeiVec3) noexcept;

		/**
		 * @brief Specify storage for a multisample texture.
		 * @tparam Dim The dimension of the texture. 1D multisampled texture is not supported and hence should not be used.
		 * @param samples Specify the number of samples in the texture.
		 * @param internal Specifies the sized internal format to be used to store texture image data.
		 * @param dimension Specifies the width, height, depth of the texture, in texels.
		 * OpenGL does not support 1D multisampled texture.
		 * For 2D texture, z component is ignored.
		 * @param fixed Specifies whether the image will use identical sample locations and the same number of samples for all texels in the image, 
		 * and the sample locations will not depend on the internal format or size of the image. 
		*/
		void textureStorageMultisample2D(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPGLVector::STPsizeiVec2, STPOpenGL::STPboolean) noexcept;
		void textureStorageMultisample3D(STPOpenGL::STPsizei, STPOpenGL::STPenum, STPGLVector::STPsizeiVec3, STPOpenGL::STPboolean) noexcept;

		/**
		 * @brief Specify a three-dimensional texture subimage.
		 * @tparam Dim The dimension of the texture, it reflects the glTextureSubImage function to be called.
		 * @param level Specifies the level-of-detail number. Level 0 is the base image level. Level n is the nth mipmap reduction image.
		 * @param offset Specifies a texel offset in the x, y, z direction within the texture array.
		 * For 1D texture, y and z offset are ignored.
		 * For 2D or 1D array texture, z offset is ignored.
		 * @param dimension Specifies the width, height, depth of the texture subimage.
		 * The dimension requirement remains the same as textureStorage() function.
		 * @param format Specifies the format of the pixel data.
		 * @param type Specifies the data type of the pixel data.
		 * @param pixel Specifies a pointer to the image data in memory.
		*/
		void textureSubImage1D(STPOpenGL::STPint, STPOpenGL::STPint, STPOpenGL::STPsizei, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*) noexcept;
		void textureSubImage2D(STPOpenGL::STPint, STPGLVector::STPintVec2, STPGLVector::STPsizeiVec2, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*) noexcept;
		void textureSubImage3D(STPOpenGL::STPint, STPGLVector::STPintVec3, STPGLVector::STPsizeiVec3, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*) noexcept;

		/**
		 * @brief Return a texture image.
		 * @param level Specifies the level-of-detail number of the desired image.
		 * Level 0 is the base image level. Level n is the nth mipmap reduction image.
		 * @param format Specifies a pixel format for the returned data.
		 * @param type Specifies a pixel type for the returned data.
		 * @param bufSize Specifies the size of the buffer pixels.
		 * @param pixel Returns the texture image. Should be a pointer to an array of the type specified by type.
		*/
		void getTextureImage(STPOpenGL::STPint, STPOpenGL::STPenum, STPOpenGL::STPenum, STPOpenGL::STPsizei, void*) const noexcept;

		/**
		 * @brief Fills all a texture image with a constant value
		 * @param level The level of texture containing the region to be cleared.
		 * @param format The format of the data whose address in memory is given by data.
		 * @param type The type of the data whose address in memory is given by data.
		 * @param data The address in memory of the data to be used to clear the specified region.
		*/
		void clearTextureImage(STPOpenGL::STPint, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*) noexcept;

	};

}
#endif//_STP_TEXTURE_H_