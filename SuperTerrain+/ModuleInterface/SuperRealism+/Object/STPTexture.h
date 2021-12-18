#pragma once
#ifndef _STP_TEXTURE_H_
#define _STP_TEXTURE_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

//GLM
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPTexture is a thin wrapper to GL texture objects and smartly handle its lifetime.
	*/
	class STP_REALISM_API STPTexture {
	private:

		/**
		 * @brief STPTextureDeleter is a smart deleter for GL texture buffer object.
		*/
		struct STP_REALISM_API STPTextureDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPTextureDeleter>> STPSmartTexture;
		//TBO
		STPSmartTexture Texture;

	public:

		//The target this texture object is bound to.
		const STPOpenGL::STPenum Target;

		/**
		 * @brief Init a new and empty STPTexture.
		 * @param target Specifies the effective texture target of each created texture.
		*/
		STPTexture(STPOpenGL::STPenum);

		STPTexture(const STPTexture&) = delete;

		STPTexture(STPTexture&&) noexcept = default;

		STPTexture& operator=(const STPTexture&) = delete;

		STPTexture& operator=(STPTexture&&) noexcept = default;

		~STPTexture() = default;

		/**
		 * @brief Get the underlying texture buffer object.
		 * @return The texture object.
		*/
		STPOpenGL::STPuint operator*() const;

		/**
		 * @brief Bind an existing texture object to the specified texture unit 
		 * @param unit Specifies the texture unit, to which the texture object should be bound to. 
		*/
		void bind(STPOpenGL::STPuint) const;

		/**
		 * @brief Bind a level of a texture to an image unit.
		 * @param unit Specifies the index of the image unit to which to bind the texture.
		 * @param level Specifies the level of the texture that is to be bound.
		 * @param layered Specifies whether a layered texture binding is to be established.
		 * @param layer If layered is GL_FALSE, specifies the layer of texture to be bound to the image unit. Ignored otherwise.
		 * @param access Specifies a token indicating the type of access that will be performed on the image.
		 * @param format Specifies the format that the elements of the image will be treated as for the purposes of formatted stores.
		*/
		void bindImage(STPOpenGL::STPuint, STPOpenGL::STPint, STPOpenGL::STPboolean, STPOpenGL::STPint, STPOpenGL::STPenum, STPOpenGL::STPenum) const;

		/**
		 * @brief Unbine a texture from the image unit.
		 * @param unit The texture unit to be unbound.
		*/
		static void unbindImage(STPOpenGL::STPuint);

		/**
		 * @brief Unbind a texture unit.
		 * @param unit The unit to be unbound.
		*/
		static void unbind(STPOpenGL::STPuint);

		/**
		 * @brief Generate mipmaps for a specified texture object
		*/
		void generateMipmap();

		/**
		 * @brief Set the texture filtering.
		 * @param min The filter mode for minification.
		 * @param mag The filter mode for magnification.
		*/
		void filter(STPOpenGL::STPenum, STPOpenGL::STPenum);

		/**
		 * @brief Set the texture wrap mode.
		 * @param s The texture warp mode for X direction.
		 * @param t The texture warp mode for Y direction.
		 * @param r The texture warp mode for Z direction.
		*/
		void wrap(STPOpenGL::STPenum, STPOpenGL::STPenum, STPOpenGL::STPenum);

		/**
		 * @brief Set the same texture wrap mode for all direction.
		 * @param str The texture warp mode for XYZ direction.
		*/
		void wrap(STPOpenGL::STPenum);

		/**
		 * @brief Set the border color when texture is wrapped using border mode.
		 * @param color The border color.
		*/
		void borderColor(glm::vec4);

		/**
		 * @brief Set the anisotropy filtering mode for the texture.
		 * @param ani The filter level.
		*/
		void anisotropy(STPOpenGL::STPfloat);

		/**
		 * @brief Allocate immutable storage for a texture.
		 * @param level Specify the number of texture levels.
		 * @param internal Specifies the sized internal format to be used to store texture image data.
		 * @param dimension Specifies the width, height, depth of the texture, in texels. 
		 * For 1D texture, y and z component should be 1.
		 * For 2D texture, z component should be 1.
		 * None of the componet should be 0.
		*/
		void textureStorage(STPOpenGL::STPint, STPOpenGL::STPenum, glm::uvec3);

		/**
		 * @brief Specify a three-dimensional texture subimage.
		 * @param level Specifies the level-of-detail number. Level 0 is the base image level. Level n is the nth mipmap reduction image.
		 * @param offset Specifies a texel offset in the x, y, z direction within the texture array.
		 * For 1D texture, y and z offset are ignored.
		 * For 2D texture, z offset is ignored.
		 * @param dimension Specifies the width, height, depth of the texture subimage.
		 * The dimension requirment remains the same as textureStorage() function.
		 * @param format Specifies the format of the pixel data.
		 * @param type Specifies the data type of the pixel data.
		 * @param pixel Specifies a pointer to the image data in memory.
		*/
		void textureSubImage(STPOpenGL::STPint, glm::ivec3, glm::uvec3, STPOpenGL::STPenum, STPOpenGL::STPenum, const void*);

	};

}
#endif//_STP_TEXTURE_H_