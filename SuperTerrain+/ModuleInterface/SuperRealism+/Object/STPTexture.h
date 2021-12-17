#pragma once
#ifndef _STP_TEXTURE_H_
#define _STP_TEXTURE_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

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

	};

}
#endif//_STP_TEXTURE_H_