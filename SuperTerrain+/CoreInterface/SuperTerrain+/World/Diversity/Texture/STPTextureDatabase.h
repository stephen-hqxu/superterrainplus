#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <vector>
#include <utility>
#include <tuple>
//System
#include <memory>

//GLM
#include <glm/vec2.hpp>
//GL
#include <SuperTerrain+/STPOpenGL.h>

//Texture
#include "STPTextureInformation.hpp"
#include "STPTextureType.hpp"

namespace SuperTerrainPlus::STPDiversity {

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

			//The dimension of the texture
			glm::uvec2 Dimension;
			//The internal format of the texture during memory allocation
			STPOpenGL::STPenum InteralFormat;
			//The channel format of each image in the group during memory transaction
			STPOpenGL::STPenum ChannelFormat;
			//The format of each pixel in each image
			STPOpenGL::STPenum PixelFormat;

		};

		/**
		 * @brief STPTextureSplatBuilder is a simple utility that allows each biome to have different texture.
		 * Texture with a biome can be arranged either by altitude or gradient at any point on the terrain mesh.
		*/
		class STP_API STPTextureSplatBuilder {
		private:

			friend class STPTextureDatabase;

			//A database from the parent texture database
			STPTextureDatabase::STPTextureDatabaseImpl* Database;

			/**
			 * @brief Init STPTextureSplatBuilder.
			 * @database The pointer to the texture database
			*/
			STPTextureSplatBuilder(STPTextureDatabase::STPTextureDatabaseImpl*);

			STPTextureSplatBuilder(const STPTextureSplatBuilder&) = delete;

			STPTextureSplatBuilder(STPTextureSplatBuilder&&) noexcept;

			STPTextureSplatBuilder& operator=(const STPTextureSplatBuilder&) = delete;

			STPTextureSplatBuilder& operator=(STPTextureSplatBuilder&&) noexcept;

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

		public:

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
			 * @brief Get the number of altitude structure in the table
			 * @return The number of altitude structure
			*/
			size_t altitudeSize() const;

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

			/**
			 * @brief Get the number of gradient structure in the table
			 * @return The number of gradient structure
			*/
			size_t gradientSize() const;

		};

		/**
		 * @brief STPDatabaseView is a visitor to a database instance, and allows quering large result sets from database.
		*/
		class STP_API STPDatabaseView {
		public:

			//A database from the parent texture database
			const STPTextureDatabase& Database;

		private:

			//The pointer from the database, store them just for convenience
			STPTextureDatabase::STPTextureDatabaseImpl* const Impl;
			const STPTextureDatabase::STPTextureSplatBuilder& SplatBuilder;

		public:

			//A result set contains sample to each splat configuration mapping
			template<class N>
			using STPNodeRecord = std::vector<std::pair<Sample, N>>;
			typedef STPNodeRecord<STPTextureInformation::STPAltitudeNode> STPAltitudeRecord;
			typedef STPNodeRecord<STPTextureInformation::STPGradientNode> STPGradientRecord;
			typedef std::vector<std::tuple<Sample, size_t, size_t>> STPSampleRecord;

			//A vector contains group ID with corresponding group properties
			typedef std::vector<std::tuple<
				STPTextureInformation::STPTextureGroupID,
				//The number of texture data in this group
				size_t,
				STPTextureDescription
			>> STPGroupRecord;
			//A vector of texture ID
			typedef std::vector<STPTextureInformation::STPTextureID> STPTextureRecord;
			//A vector contains texture data
			typedef std::vector<std::tuple<
				STPTextureInformation::STPTextureGroupID,
				STPTextureInformation::STPTextureID,
				STPTextureType,
				const void*
			>> STPMapRecord;
			//A vector of texture type
			typedef std::vector<STPTextureType> STPTextureTypeRecord;

			/**
			 * @brief Init STPDatabaseView with a database instance
			 * @param db The pointer to the database. Note that the database view is a non-owning object of database, all copy operations will be shallow.
			*/
			STPDatabaseView(const STPTextureDatabase&);

			STPDatabaseView(const STPDatabaseView&) = default;

			STPDatabaseView(STPDatabaseView&&) = default;

			STPDatabaseView& operator=(const STPDatabaseView&) = default;

			STPDatabaseView& operator=(STPDatabaseView&&) = default;

			~STPDatabaseView() = default;

			/* -- Splat Builder --*/

			/**
			 * @brief Retrieve all stored altitude configurations.
			 * Altitude records will be sorted by sample; if samples are the same it will be then sorted by upper bound
			 * @return The sorted altitude record.
			*/
			STPAltitudeRecord getAltitudes() const;

			/**
			 * @brief Retrieve all stored gradient configurations.
			 * Gradient records will be sorted by sample.
			 * @return The sorted gradient record.
			*/
			STPGradientRecord getGradients() const;


			/**
			 * @brief Get an array of samples that have been registered with any splat configuration.
			 * Also return the number of rule each sample owns.
			 * @param hint An optional prediction of the number of possible presented sample to speed up memory allocation
			 * @return An array sample with any splat configuration.
			*/
			STPSampleRecord getValidSample(unsigned int = 0u) const;

			/* -- Database -- */

			/**
			 * @brief Retrieve a record of all groups that have been referenced by some texture data, and their properties.
			 * Note that group only being added to the database but not referenced by any splat rule does not count.
			 * Results are sorted by group ID in ascending order.
			 * @return An array of group record.
			*/
			STPGroupRecord getValidGroup() const;

			/**
			 * @brief Retrieve a record of all texture ID in this database.
			 * Result will be sorted in ascending order.
			 * Any texture (represented by texture ID) with no splat rule referenced to it will be ignored.
			 * @return An array of sorted texture ID
			*/
			STPTextureRecord getValidTexture() const;

			/**
			 * @brief Retrieve a record of all texture map in this database.
			 * Result will be sorted in ascending order of texture group ID.
			 * Only map from the texture which is valid will be included.
			 * @return An array of map sorted by group ID
			*/
			STPMapRecord getValidMap() const;

			/**
			 * @brief Retrieve a record of all texture map type being used in this database.
			 * Type will be sorted by their defined numeric value, which can be used as indices.
			 * @param hint An optional prediction on how many types will be used, this can be used to speed up memory allocation.
			 * @return An array of sorted texture type.
			*/
			STPTextureTypeRecord getValidMapType(unsigned int = 0u) const;

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

		/**
		 * @brief Expand parameter packs for addMaps() template function and group parameters into callable arguments for non-template function.
		 * TODO: make this a lambda template in C++20
		 * @tparam ...Arg All parameters passed as parameter pack
		 * @param texture_id The texture ID to be operated on
		 * @param Striding index sequence to index the tuple
		 * @param args The argument to be expanded.
		*/
		template<size_t... Is, class... Arg>
		void expandAddMaps(STPTextureInformation::STPTextureID, std::index_sequence<Is...>, std::tuple<Arg...>);

	public:

		/**
		 * @brief Init an empty texture database
		*/
		STPTextureDatabase();

		STPTextureDatabase(const STPTextureDatabase&) = delete;

		STPTextureDatabase(STPTextureDatabase&&) noexcept;

		STPTextureDatabase& operator=(const STPTextureDatabase&) = delete;

		STPTextureDatabase& operator=(STPTextureDatabase&&) noexcept;

		~STPTextureDatabase();

		/**
		 * @brief Get the pointer to splat builder to configure terrain splating.
		 * @return The pointer to splat builder managed by the texture database
		*/
		STPTextureSplatBuilder& getSplatBuilder();

		/**
		 * @brief Get the pointer to splat builder to configure terrain splating.
		 * @return The pointer to splat builder managed by the texture database
		*/
		const STPTextureSplatBuilder& getSplatBuilder() const;

		/**
		 * @brief Get the database view
		 * @return The database view object
		*/
		STPDatabaseView visit() const;

		/**
		 * @brief Insert a new texture group into the texture database
		 * @param desc The texture format description, it will be applied to all texture data in this group
		 * @return The group ID of the newly inserted texture.
		 * A group will always be inserted since group ID is managed by the database and guaranteed to be unique
		*/
		STPTextureInformation::STPTextureGroupID addGroup(const STPTextureDescription&);

		/**
		 * @brief Remove a texture group from the texture database, all texture maps in the group will be dropped.
		 * No operation will be performed if group ID is invalid.
		 * @param group_id The group with ID to be removed
		*/
		void removeGroup(STPTextureInformation::STPTextureGroupID);

		/**
		 * @brief Get the pointer to the texture group, given a group ID
		 * @param id The texture group ID
		 * @return The group description for requested group
		 * If group ID is not found, exception is thrown.
		*/
		STPTextureDescription getGroupDescription(STPTextureInformation::STPTextureGroupID) const;

		/**
		 * @brief Get the number of texture group registered
		 * @return The number of registered texture group
		*/
		size_t groupSize() const;

		/**
		 * @brief Insert a new texture into texture database. New texture has no map, and can be added by calling addMap().
		 * A texture may have a collection of different type of maps associated with the texture.
		 * @return The texture ID(s) that can be used to reference the texture
		*/
		STPTextureInformation::STPTextureID addTexture();

		/**
		 * @brief Insert a few new texture containers into the texture database. New texture has no map, and can be added by calling addMap().
		 * A texture may have a collection of different type of maps associated with the texture.
		 * @param count The number of texture ID to be added.
		 * @param texture_id The pointer to the returning texture ID, must have the same of greater size than count.
		*/
		void addTexture(unsigned int, STPTextureInformation::STPTextureID*);

		/**
		 * @brief Remove a texture from the texture database. 
		 * All texture maps owned by this texture will be dropped, all splat rules using the texture ID will be dropped.
		 * No operation will be performed if texture ID is invalid
		 * @param texture_id The texture ID to be removed
		*/
		void removeTexture(STPTextureInformation::STPTextureID);

		/**
		 * @brief Add a new map to the texture database for a particular texture.
		 * @param texture_id The ID of the texture to be added with map.
		 * @param type The type of the map for this texture, to identify a specific texture for a texture ID
		 * @param group_id The ID of the texture group. All texture in the same group must have the same texture description
		 * @param texture_data The pointer to the texture data. Texture data is not owned by the database, thus user should guarantees the lifetime;
		 * the texture data should match the property of the group when it was created it.
		*/
		void addMap(STPTextureInformation::STPTextureID, STPTextureType, STPTextureInformation::STPTextureGroupID, const void*);

		/**
		 * @brief For a texture, add a sequence of texture map with different types to the texture to the texture database.
		 * If any of the insert texture data operation fails, exception will be thrown immediately and subsequent operations will be halted.
		 * @tparam ...Ret A tuple of booleans to indicate insert status
		 * @tparam ...Arg Sequence of arguments for adding texture
		 * @param texture_id The texture ID to be added
		 * @param ...args A sequence of argument packs to add the texture data. See the non-template version of addMap()
		 * The last 3 arguments can be repeated.
		*/
		template<class... Arg>
		void addMaps(STPTextureInformation::STPTextureID, Arg&&...);

		/**
		 * @brief Get the number of texture map, that contains all texture with all types, in the database.
		 * @return The number of texture map.
		*/
		size_t mapSize() const;

		/**
		 * @brief Get the number of texture registered.
		 * Note that one texture may contain multiple maps of different types associated to different groups
		 * @return The number of registered texture
		*/
		size_t textureSize() const;

	};

}
#include "STPTextureDatabase.inl"
#endif//_STP_TEXTURE_DATABASE_H_