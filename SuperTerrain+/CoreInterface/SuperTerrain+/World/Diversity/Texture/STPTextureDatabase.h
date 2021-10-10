#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <array>
#include <vector>
#include <unordered_map>
#include <utility>
//System
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
			typedef unsigned int STPTextureID;
			//Each group has an ID to uniquely identify a texture group in the database
			typedef unsigned int STPTextureGroupID;

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

		private:

			//ID counter
			static size_t IDAccumulator;

			//Given a texture type, check if the texture type is associated with a data
			typedef std::unordered_map<STPTextureType, std::pair<const void*, STPTextureGroupID>> STPTypeInformation;

			//Given a textuer ID, find all texture types related to this ID as well as the group ID where this type of textuer is in
			std::unordered_map<STPTextureID, STPTypeInformation> TextureTypeMapping;
			//All texture groups owned by the database
			std::unordered_map<STPTextureGroupID, STPTextureDescription> TextureGroupRecord;

			//An array of non-owning data structure contains texture information
			template<typename ID, class S>
			using STPTextureDataView = std::vector<std::pair<ID, const S*>>;

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

			/**
			 * @brief A universal implementation to sort a texture data and put them into an array of non-owning view
			 * @tparam ID The ID type
			 * @tparam S The texture structure type of the output
			 * @tparam M The texture structure type of the input
			 * @param mapping The selected mapping to be sorted.
			 * @return A sorted array aginsted ID value and pointer to each element in mapping
			*/
			template<typename ID, class S, class M>
			static STPTextureDataView<ID, S> sortView(const M&);

		public:

			//An array of non-owning texture ID to type group mapping
			typedef STPTextureDataView<STPTextureID, STPTypeInformation> STPTypeMappingView;
			//An array of non-owning texture group record
			typedef STPTextureDataView<STPTextureGroupID, STPTextureDescription> STPGroupView;

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
			const STPTypeInformation& getTypeMapping(STPTextureID) const;

			/**
			 * @brief Sort the texture type-groupID mapping based on texture ID.
			 * This can be used to convert texture ID to index a texture mapping in an array.
			 * @return A vector of texture ID and non-owning pointer to type group mapping.
			 * Note that state of pointer may change and undefined if texture database is modified after this function returns
			*/
			STPTypeMappingView sortTypeMapping() const;

			/**
			 * @brief Get the pointer to the texture group, given a group ID
			 * @param id The texture group ID
			 * @return The pointer to the group with that group ID.
			 * If group ID is not found, exception is thrown.
			*/
			const STPTextureDescription& getGroupDescription(STPTextureGroupID) const;

			/**
			 * @brief Sort the texture group record based on group ID.
			 * This can be used to convert group ID to index a group in an array
			 * @return A vector of group ID and non-owning pointer to group.
			 * Note that the state of pointer may change and undefined if texture database is modified after this function returns
			*/
			STPGroupView sortGroup() const;

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
			STPTextureGroupID addGroup(const STPTextureDescription&);

			/**
			 * @brief Insert a new texture into the texture database. New texture has no content, and can be added by calling addTextureData().
			 * A texture may have a collection of different types associated with the texture.
			 * @return The texture ID that can be used to reference the texture
			*/
			STPTextureID addTexture();

			/**
			 * @brief Add a new texture data to the texture database for a particular texture
			 * @param texture_id The ID of the texture to be added with data.
			 * @param type The type of the texture, to identify a specific texture for a texture ID
			 * @param group_id The ID of the texture group. All texture in the same group must have the same texture description
			 * @param texture_data The pointer to the texture data. Texture data is not owned by the database, thus user should guarantees the lifetime; 
			 * the texture data should match the property of the group when it was created it.
			 * @return True if texture has been inserted into the database
			 * False if texture ID cannot be found, or the texture group ID cannot be found.
			*/
			bool addTextureData(STPTextureID, STPTextureType, STPTextureGroupID, const void*);

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
			auto addTextureDatas(STPTextureID, Arg&&...);

		};

	}
}
#include "STPTextureDatabase.inl"
#endif//_STP_TEXTURE_DATABASE_H_