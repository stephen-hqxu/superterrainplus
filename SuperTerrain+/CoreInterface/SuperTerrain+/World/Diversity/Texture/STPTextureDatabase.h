#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <vector>
#include <utility>
//System
#include <memory>

//GLM
#include <glm/vec2.hpp>
//GLAD
#include <glad/glad.h>

#include "STPTextureInformation.hpp"

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
		private:

			//implementation of texture database
			class STPTextureDatabaseImpl;

		public:

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
			 * @brief STPTextureSplatBuilder is a simple utility that allows each biome to have different texture.
			 * Texture with a biome can be arranged either by altitude or gradient at any point on the terrain mesh.
			*/
			class STP_API STPTextureSplatBuilder {
			private:

				friend class STPTextureDatabase;

				//an array of non-owning biome structure
				template<class S>
				using STPStructureView = std::vector<std::pair<Sample, const S*>>;

			public:

				//array of non-owning biome altitude structure
				typedef STPStructureView<STPTextureInformation::STPAltitudeNode> STPAltitudeView;
				//array of non-owning biome gradient structure
				typedef STPStructureView<STPTextureInformation::STPGradientNode> STPGradientView;

			private:

				//A database from the parent texture database
				STPTextureDatabase::STPTextureDatabaseImpl* const Database;

				/**
				 * @brief Init STPTextureSplatBuilder.
				 * @database The pointer to the texture database
				*/
				STPTextureSplatBuilder(STPTextureDatabase::STPTextureDatabaseImpl*);

				STPTextureSplatBuilder(const STPTextureSplatBuilder&) = delete;

				STPTextureSplatBuilder(STPTextureSplatBuilder&&) = delete;

				STPTextureSplatBuilder& operator=(const STPTextureSplatBuilder&) = delete;

				STPTextureSplatBuilder& operator=(STPTextureSplatBuilder&&) = delete;

				~STPTextureSplatBuilder() = default;

				//TODO: template lambda...
				/**
				 * @brief Expand tuple of arguments and add each group into the configuration
				 * @tparam ...Arg Arguments that are packed into tuple
				 * @param sample The sample to be operated
				 * @param Indices to get the arguments from tuple
				 * @param args All arguments packed
				*/
				template<size_t... Is, class... Arg>
				void expandAddAltitudes(Sample, std::index_sequence<Is...>, std::tuple<Arg...>);

				/**
				 * @brief Expand tuple of arguments and add each group into the configurations
				 * @tparam ...Arg Arguments that are packed into tuple
				 * @param sample The sample to be operated
				 * @param Indices to get the arguments from tuple
				 * @param args All arguments packed
				*/
				template<size_t... Is, class... Arg>
				void expandAddGradients(Sample, std::index_sequence<Is...>, std::tuple<Arg...>);

				/**
				 * @brief Get a view to the biome mapping and return it as a non-owning vector
				 * @tparam View The type of view to be returned
				 * @tparam Struct The type of biome structure
				 * @tparam Mapping Mapping type to be sorted
				 * @param mapping The mapping to be sorted
				 * @return A vector of biome structure mapping
				*/
				template<class S, class M>
				static STPStructureView<S> visitMapping(const M&);

			public:

				/**
				 * @brief Get pointer to altitude structure for the specified sample
				 * @param sample The altitude for the sample to be retrieved
				 * @return The pointer to the altitude structure for the sample.
				 * If no altitude is associated with said sample, exception is thrown
				*/
				//const STPAltitudeStructure& getAltitude(Sample) const;

				/**
				 * @brief Visit the altitude configuration registered with the current splat builder
				 * @return A vector of read-only pointer to altitude mapping.
				 * The vector contains pointer to altitude structure which is not owning, state might be changed if more altitudes are added later.
				 * The returned vector is best suited for traversal, individual lookup may not be efficient since unsorted.
				*/
				STPAltitudeView visitAltitude() const;

				/**
				 * @brief Get the pointer to the gradient structure for the specified sample
				 * @param sample The gradient for the sample to be retrieved
				 * @return The pointer to the gradient structure for the sample
				 * If no altitude is associated with said sample, exception is thrown
				*/
				//const STPGradientStructure& getGradient(Sample) const;

				/**
				 * @brief Visit the gradient configuration registered with the current splat builder
				 * @return A vector of read-only pointer to gradient mapping.
				 * The vector contains pointer to gradient structure which is not owning, state might be chanegd if more gradients are added later.
				 * The returned vector is best suited for traversal, individual lookup may not be efficient since unsorted.
				*/
				STPGradientView visitGradient() const;

				/**
				 * @brief Add a new configuration for specified biome into altitude structure.
				 * If the upper bound already exists, the old texture ID will be replaced.
				 * @param sample The sample that the new altitude configuration belongs to
				 * @param upperBound The upper limit of altitude the region will be active
				 * @param texture_id The texture to be used in this region
				*/
				void addAltitude(Sample, float, STPTextureInformation::STPTextureID);

				/**
				 * @brief Add a set of new configurations for specified biome into altitude structure
				 * @tparam ...Arg Argument pack to be added. Must follow the argument structure of addAltitude() function
				 * @param sample The sample that the new altitude configurations belong to
				 * @param ...args Arguments grouped to be added
				*/
				template<class... Arg>
				void addAltitudes(Sample, Arg&&...);

				/**
				 * @brief Add a new configuration for specified biome into gradient structure
				 * @param sample The sample that the new gradient configuration belongs to
				 * @param minGradient Region starts from gradient higher than this value
				 * @param maxGradient Region ends with gradient lower than this value
				 * @param lowerBound Region starts from altitude higher than this value
				 * @param upperBound Region ends with altitude lower than this value
				 * @param texture_id The texture ID to be used in this region
				*/
				void addGradient(Sample, float, float, float, float, STPTextureInformation::STPTextureID);

				/**
				 * @brief Add a set of new configurations for specified biome into gradient structure
				 * @tparam ...Arg Argument pack to be added. Must follow the argument structure of addGradient() function
				 * @param sample The sample that the new gradient configurations belong to
				 * @param ...args Arguments grouped to be addded
				*/
				template<class... Arg>
				void addGradients(Sample, Arg&&...);

			};

		private:

			//incremental accumulators for ID
			inline static unsigned int GeneralIDAccumulator = 10000u;
			inline static STPTextureInformation::STPTextureID TextureIDAccumulator = 10000u;
			inline static STPTextureInformation::STPTextureGroupID GroupIDAccumulator = 9999u;

			//A database which stores all biome texture settings
			std::unique_ptr<STPTextureDatabaseImpl> Database;
			//implementations that depend on the database
			STPTextureSplatBuilder SplatBuilder;

			//An array of non-owning data structure contains texture information
			template<typename ID, class S>
			using STPTextureDataView = std::vector<std::pair<ID, const S*>>;

			/**
			 * @brief Expand parameter packs for addTextures() template function and group parameters into callable arguments for non-template function.
			 * TODO: make this a lambda template in C++20
			 * @tparam ...Arg All parameters passed as parameter pack
			 * @param texture_id The texture ID to be operated on
			 * @param Striding index sequence to index the tuple
			 * @param args The argument to be expanded.
			*/
			template<size_t... Is, class... Arg>
			void expandAddTextures(STPTextureInformation::STPTextureID, std::index_sequence<Is...>, std::tuple<Arg...>);

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
			//typedef STPTextureDataView<STPTextureID, STPTypeInformation> STPTypeMappingView;
			//An array of non-owning texture group record
			typedef STPTextureDataView<STPTextureInformation::STPTextureGroupID, STPTextureDescription> STPGroupView;

			/**
			 * @brief Init an empty texture database
			*/
			STPTextureDatabase();

			STPTextureDatabase(const STPTextureDatabase&) = delete;

			STPTextureDatabase(STPTextureDatabase&&) = delete;

			STPTextureDatabase& operator=(const STPTextureDatabase&) = delete;

			STPTextureDatabase& operator=(STPTextureDatabase&&) = delete;

			~STPTextureDatabase();

			/**
			 * @brief Get the pointer to the texture type-groupID mapping.
			 * That is, given a texture type, return the group ID the texture type with this texture ID it's in
			 * @param id The texture ID to be retrieved
			 * @return The pointer to the mapping
			*/
			//const STPTypeInformation& getTypeMapping(STPTextureID) const;

			/**
			 * @brief Sort the texture type-groupID mapping based on texture ID.
			 * This can be used to convert texture ID to index a texture mapping in an array.
			 * @return A vector of texture ID and non-owning pointer to type group mapping.
			 * Note that state of pointer may change and undefined if texture database is modified after this function returns
			*/
			//STPTypeMappingView sortTypeMapping() const;

			/**
			 * @brief Get the pointer to the texture group, given a group ID
			 * @param id The texture group ID
			 * @return The pointer to the group with that group ID.
			 * If group ID is not found, exception is thrown.
			*/
			const STPTextureDescription& getGroupDescription(STPTextureInformation::STPTextureGroupID) const;

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
			const void* operator()(STPTextureInformation::STPTextureID, STPTextureType) const;

			/**
			 * @brief Insert a new texture group into the texture database
			 * @param desc The texture format description, it will be applied to all texture data in this group
			 * @return The group ID of the newly inserted texture.
			 * A group will always be inserted since group ID is managed by the database and guaranteed to be unique
			*/
			STPTextureInformation::STPTextureGroupID addGroup(const STPTextureDescription&);

			/**
			 * @brief Insert a new texture into the texture database. New texture has no content, and can be added by calling addTextureData().
			 * A texture may have a collection of different types associated with the texture.
			 * @return The texture ID that can be used to reference the texture
			*/
			STPTextureInformation::STPTextureID addTexture();

			/**
			 * @brief Add a new texture data to the texture database for a particular texture.
			 * Exception will be thrown if insertion fails.
			 * @param texture_id The ID of the texture to be added with data.
			 * @param type The type of the texture, to identify a specific texture for a texture ID
			 * @param group_id The ID of the texture group. All texture in the same group must have the same texture description
			 * @param texture_data The pointer to the texture data. Texture data is not owned by the database, thus user should guarantees the lifetime; 
			 * the texture data should match the property of the group when it was created it.
			*/
			void addTextureData(STPTextureInformation::STPTextureID, STPTextureType, STPTextureInformation::STPTextureGroupID, const void*);

			/**
			 * @brief For a texture ID, add a sequence of texture data with different types to the texture to the texture database.
			 * If any of the insert texture data operation fails, exception will be thrown immediately and subsequent operations will be halted.
			 * @tparam ...Ret A tuple of booleans to indicate insert status
			 * @tparam ...Arg Sequence of arguments for adding texture
			 * @param texture_id The texture ID to be added
			 * @param ...args A sequence of argument packs to add the texture data. See the non-template version of addTexture()
			 * The last 3 arguments can be repeated.
			*/
			template<class... Arg>
			void addTextureDatas(STPTextureInformation::STPTextureID, Arg&&...);

		};

	}
}
#include "STPTextureDatabase.inl"
#endif//_STP_TEXTURE_DATABASE_H_