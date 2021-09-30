#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <unordered_map>
#include <array>
#include <utility>
//System
#include <string>
#include <tuple>
#include <type_traits>

//GLM
#include <glm/vec2.hpp>
//GLAD
#include <glad/glad.h>

#include "STPTextureType.hpp"

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
			typedef std::unordered_map<STPTextureType, STPTextureGroup::STPTextureGroupID> STPTypeGroupMapping;

			//All textuer groups owned by the database
			std::unordered_map<STPTextureGroup::STPTextureGroupID, STPTextureGroup> TextureGroupRecord;
			//Given a textuer ID, find all textuer types related to this ID as well as the group ID where this type of textuer is in
			std::unordered_map<STPTextureID, STPTypeGroupMapping> TextureTypeMapping;

			/**
			 * @brief Expand parameter packs for addTextures() template function and group parameters into callable arguments for non-template function
			 * TODO: make this a lambda template in C++20
			 * @tparam ...Arg All parameters passed as parameter pack
			 * @param texture_id The texture ID to be operated on
			 * @param Striding index sequence to index the tuple
			 * @param args The argument to be expanded
			 * @return An array of bool denoting the status
			*/
			template<size_t... Is, class... Arg>
			auto expandAddTextures(STPTextureID, std::index_sequence<Is...>, std::tuple<Arg...>);

		public:

			/**
			 * @brief Init an empty texture database
			*/
			STPTextureDatabase() = default;

			STPTextureDatabase(const STPTextureDatabase&) = delete;

			STPTextureDatabase(STPTextureDatabase&&) noexcept = default;

			STPTextureDatabase& operator=(const STPTextureDatabase&) = delete;

			STPTextureDatabase& operator=(STPTextureDatabase&&) noexcept = default;

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
			 * @brief Add a new texture data to the texture database
			 * @param texture_id The ID of the texture, it may point to multiple texture of different types
			 * @param type The type of the texture, to identify a specific texture for a texture ID
			 * @param group_id The ID of the texture group. All texture in the same group must have the same texture description
			 * @param texture_data The pointer to the texture data. Texture data is not owned by the database, thus user should guarantees the lifetime; 
			 * the texture data should match the property of the group when it was created it.
			 * @return True if texture has been inserted into the database
			 * False the pair of texture ID and texture type is not unique in the database, or the texture group ID cannot be found.
			*/
			bool addTexture(STPTextureID, STPTextureType, STPTextureGroup::STPTextureGroupID, const void*);

			/**
			 * @brief For a texture ID, add a sequence of texture data with different types to the texture to the texture database
			 * @tparam ...Ret A tuple of booleans to indicate insert status
			 * @tparam ...Arg Sequence of arguments for adding texture
			 * @param texture_id The texture ID to be added
			 * @param ...args A sequence of argument packs to add the texture data. See the non-template version of addTexture()
			 * The last 3 arguments can be repeated
			 * @return An array of insertion status in order.
			 * False for an element if insertion did not take place for the parameter group
			*/
			template<class... Arg>
			auto addTextures(STPTextureID, Arg&&...);

		};

	}
}
#include "STPTextureDatabase.inl"
#endif//_STP_TEXTURE_DATABASE_H_