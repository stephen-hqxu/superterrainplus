#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <map>
#include <unordered_map>
#include <utility>
//System
#include <string>

//GLM
#include <glm/vec2.hpp>
//GLAD
#include <glad/glad.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPDiversity is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPDiversity {

		/**
		 * @brief STPTextureDatabase is a non-owning storage of collection of texture data and information.
		 * Texture is identified in the database with a unique ID. Each texture ID contains a collection of texture types, like albedo map and normal map.
		 * All maps with the same texture format will be packed into groups, internally.
		 * Given a texture ID, STPTextureDatabase is able to retrieve all different map types registered previously.
		*/
		class STP_API STPTextureDatabase {
		public:

			//Each texture collection has an string ID to uniquely identify a texture with different types in the database
			typedef std::string STPTextureID;

			/**
			 * @brief STPTextureType defines the type of texture the texture group holds
			*/
			enum class STPTextureType : unsigned char {
				//A texture that defines the base color of the mesh being textured
				Albedo = 0x00u,
				//A texture that defines the normal vector which is then used to calculate light reflection and refraction on the surface of the mesh
				Normal = 0x01u,
				//A texture that defines the perpendicular offset to the surface of the mesh at a pixel
				Displacement = 0x02u,
				//A texture that defines the amount of specular highlight at a pixel
				Specular = 0x03u,
				//A texture that defines how much light is scattered across the surface of the mesh
				Glossiness = 0x10u,
				//A texture that controls how much color from the albedo map contributes to the diffuse and brightness
				Metalness = 0x11u,
				//A texture that defines how a texture reacts to light during rendering
				AmbientOcclusion = 0x12u,
				//A texture that defines which part of the object emits light, as well as the light color of each pixel
				Emissive = 0x13u
			};

			/**
			 * @brief STPTextureDescription contains information about a texture
			*/
			struct STPTextureDescription {
			public:

				//Define the channel format, only supports format supported by OpenGL standard
				typedef GLenum STPChannelFormat;

				//The dimension of the texture
				glm::uvec2 Dimension;
				//The format of the texture, see OpenGL documentation for all supporting channel formats
				STPChannelFormat Format;

			};

			/**
			 * @brief STPTextureGroup is a collection of texture data with the same property
			*/
			class STP_API STPTextureGroup {
			public:

				//Each group has an ID to uniquely identify a texture group in the database
				typedef unsigned int STPTextureGroupID;
				//A pair of texture ID and texture type to form a primary key to locate the texture data in the group
				typedef std::pair<STPTextureID, STPTextureType> STPTextureKey;

			private:

				friend class STPTextureDatabase;

				//A group ID counter, behaves the same as a texture ID
				static STPTextureGroupID ReferenceAccumulator;

				/**
				 * @brief A hasher to hash a texture key
				*/
				struct STPKeyHasher {
				public:

					size_t operator()(const STPTextureKey&) const;

				};

				std::unordered_map<STPTextureKey, const void*, STPKeyHasher> TextureDataRecord;

			public:

				//The ID of the current group in the database
				const STPTextureGroupID GroupID;
				//The property of all texture in the current group
				const STPTextureDescription TextureProperty;

				/**
				 * @brief Init a STPTextureGroup with specified texture property
				 * @param desc The property of the group. It will be applied to all texture attached to this group.
				 * Property will be copied to the group, no reference will be retained after the function returns
				*/
				STPTextureGroup(const STPTextureDescription&);

				STPTextureGroup(const STPTextureGroup&) = delete;

				STPTextureGroup(STPTextureGroup&&) = delete;

				STPTextureGroup& operator=(const STPTextureGroup&) = delete;

				STPTextureGroup& operator=(STPTextureGroup&&) = delete;

				~STPTextureGroup() = default;

				/**
				 * @brief Add a new texture record with a given texture ID
				 * @param key The primary key, must be unique in the texture group
				 * @param texture The pointer to the texture. Note that the texture group is a non-owning texture manager, 
				 * caller is responsible for their lifetime.
				 * Texture data must be a valid pointer with format defined by the current group, or it will incur undefined behaviour.
				 * @return True if the texture record is inserted.
				 * False if texture ID is not unique in the group
				*/
				bool add(STPTextureKey, const void*);

				/**
				 * @brief Retrieve the texture data associated with this texture ID in the group
				 * @param key The texture data with the primary key to be retrieved
				 * @return The pointer to the texture data.
				 * If no texture ID is found to be associated with this group, exception is thrown
				*/
				const void* operator[](STPTextureKey) const;

			};

		private:

			//Given a texture type, find the group ID associated with this texture ID with this type
			typedef std::map<STPTextureType, STPTextureGroup::STPTextureGroupID> STPTypeGroupMapping;

			//All textuer groups owned by the database
			std::unordered_map<STPTextureGroup::STPTextureGroupID, STPTextureGroup> TextureGroupRecord;
			//Given a textuer ID, find all textuer types related to this ID as well as the group ID where this type of textuer is in
			std::unordered_map<STPTextureID, STPTypeGroupMapping> TextureTypeMapping;

		public:

			/**
			 * @brief Init an empty texture database
			*/
			STPTextureDatabase() = default;

			STPTextureDatabase(const STPTextureDatabase&) = delete;

			STPTextureDatabase(STPTextureDatabase&&) = delete;

			STPTextureDatabase& operator=(const STPTextureDatabase&) = delete;

			STPTextureDatabase& operator=(STPTextureDatabase&&) = delete;

			~STPTextureDatabase() = default;

			/**
			 * @brief Get the pointer to the texture type-groupID mapping.
			 * That is, given a texture type, return the group ID the texture type with this texture ID it's in
			 * @param id The texture ID to be retrieved
			 * @return The pointer to the mapping
			*/
			const STPTypeGroupMapping& getTypeMapping(STPTextureID) const;

			/**
			 * @brief Get the pointer to the texture group, given a group ID
			 * @param id The texture group ID
			 * @return The pointer to the group with that group ID.
			 * If group ID is not found, exception is thrown.
			*/
			const STPTextureGroup& getGroup(STPTextureGroup::STPTextureGroupID) const;

			/**
			 * @brief Retrieve the texture data with specific type stored in the database
			 * @param id The ID of the texture that uniquely identifies a collection of texture with different types the database
			 * @param type The type of the texture from the collection
			 * @return The texture data with the ID and said type.
			 * If texture ID is not found, or the the texture ID contains no associated type, exception is thrown
			*/
			const void* operator()(STPTextureID, STPTextureType) const;

			/**
			 * @brief Insert a new texture group into the texture database
			 * @param desc The texture format description, it will be applied to all texture data in this group
			 * @return The group ID of the newly inserted texture.
			 * A group will always be inserted since group ID is managed by the database and guaranteed to be unique
			*/
			STPTextureGroup::STPTextureGroupID addGroup(const STPTextureDescription&);

			/**
			 * @brief Add a new texture data in the texture database
			 * @param texture_id The ID of the texture, it may point to multiple texture of different types
			 * @param type The type of the texture, to identify a specific texture for a texture ID
			 * @param group_id The ID of the texture group. All texture in the same group must have the same texture description
			 * @param texture_data The pointer to the texture data. Texture data is not owned by the database, thus user should guarantees the lifetime; 
			 * the texture data should match the property of the group when it was created it.
			 * @return True if texture has been inserted into the database
			 * False the pair of texture ID and texture type is not unique in the database, or the texture group ID cannot be found.
			*/
			bool addTexture(STPTextureID, STPTextureType, STPTextureGroup::STPTextureGroupID, const void*);

		};

	}
}
#endif//_STP_TEXTURE_DATABASE_H_