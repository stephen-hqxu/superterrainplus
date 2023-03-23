#pragma once
#ifndef _STP_TEXTURE_DATABASE_H_
#define _STP_TEXTURE_DATABASE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <vector>
#include <utility>
#include <tuple>
#include <string_view>
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
		 * @brief STPMapGroupDescription contains information about a texture map group.
		 * A texture map group holds a texture maps data with the same texture format.
		*/
		struct STPMapGroupDescription {
		public:

			//The dimension of the texture
			glm::uvec2 Dimension;
			//The total number of mipmap level for the texture
			unsigned int MipMapLevel;
			//The internal format of the texture during memory allocation
			STPOpenGL::STPenum InteralFormat;
			//The channel format of each image in the group during memory transaction
			STPOpenGL::STPenum ChannelFormat;
			//The format of each pixel in each image
			STPOpenGL::STPenum PixelFormat;

		};

		/**
		 * @brief STPViewGroupDescription contains information about a texture view group.
		 * A texture view group holds all texture with the same viewing properties.
		*/
		typedef STPTextureInformation::STPTextureView STPViewGroupDescription;

		/**
		 * @brief STPTextureSplatBuilder is a simple utility that allows each biome to have different texture.
		 * Texture with a biome can be arranged either by altitude or gradient at any point on the terrain mesh.
		*/
		class STP_API STPTextureSplatBuilder {
		private:

			friend class STPTextureDatabase;

			//A database from the parent texture database
			STPTextureDatabase::STPTextureDatabaseImpl& Database;

			/**
			 * @brief Init STPTextureSplatBuilder.
			 * @param database The pointer to the texture database.
			*/
			STPTextureSplatBuilder(STPTextureDatabase&);

			~STPTextureSplatBuilder() = default;

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
			 * @brief Get the number of gradient structure in the table
			 * @return The number of gradient structure
			*/
			size_t gradientSize() const;

		};

		/**
		 * @brief STPDatabaseView is a visitor to a database instance, and allows querying large result sets from database.
		*/
		class STP_API STPDatabaseView {
		public:

			//A database from the parent texture database
			const STPTextureDatabase& Database;

		private:

			//The pointer from the database, store them just for convenience
			STPTextureDatabase::STPTextureDatabaseImpl& Impl;
			const STPTextureDatabase::STPTextureSplatBuilder& SplatBuilder;

		public:

			//A result set contains sample to each splat configuration mapping
			template<class N>
			using STPNodeRecord = std::vector<std::pair<Sample, N>>;
			typedef STPNodeRecord<STPTextureInformation::STPAltitudeNode> STPAltitudeRecord;
			typedef STPNodeRecord<STPTextureInformation::STPGradientNode> STPGradientRecord;
			typedef std::vector<std::tuple<Sample, size_t, size_t>> STPSampleRecord;

			//A vector contains map group ID with corresponding map group properties
			typedef std::vector<std::tuple<
				STPTextureInformation::STPMapGroupID,
				//The number of texture data in this map group
				size_t,
				STPMapGroupDescription
			>> STPMapGroupRecord;

			//A vector of texture ID
			typedef std::vector<std::pair<
				STPTextureInformation::STPTextureID, 
				STPViewGroupDescription
			>> STPTextureRecord;

			//A vector contains texture data
			typedef std::vector<std::tuple<
				STPTextureInformation::STPMapGroupID,
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
			STPDatabaseView(const STPTextureDatabase&) noexcept;

			~STPDatabaseView() = default;

			/* ---------------------------- Splat Builder -------------------------------- */

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

			/* ----------------------- Database ----------------------------- */

			/**
			 * @brief Retrieve a record of all map groups that have been referenced by some texture map data, and their properties.
			 * Note that group only being added to the database but not referenced by any splat rule does not count.
			 * Results are sorted by group ID in ascending order.
			 * @return An array of map group record.
			*/
			STPMapGroupRecord getValidMapGroup() const;

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

		//A database which stores all biome texture settings
		std::unique_ptr<STPTextureDatabaseImpl> Database;
		//implementations that depend on the database
		STPTextureSplatBuilder SplatBuilder;

	public:

		/**
		 * @brief Init an empty texture database
		*/
		STPTextureDatabase();

		STPTextureDatabase(const STPTextureDatabase&) = delete;

		STPTextureDatabase(STPTextureDatabase&&) noexcept;

		STPTextureDatabase& operator=(const STPTextureDatabase&) = delete;

		STPTextureDatabase& operator=(STPTextureDatabase&&) = delete;

		~STPTextureDatabase();

		/**
		 * @brief Get the pointer to splat builder to configure terrain splatting.
		 * @return The pointer to splat builder managed by the texture database.
		*/
		STPTextureSplatBuilder& splatBuilder() noexcept;

		/**
		 * @brief Get the pointer to splat builder to configure terrain splatting.
		 * @return The pointer to splat builder managed by the texture database.
		*/
		const STPTextureSplatBuilder& splatBuilder() const noexcept;

		/**
		 * @brief Get the database view.
		 * @return The database view object.
		*/
		STPDatabaseView visit() const noexcept;

		/**
		 * @brief Insert a new texture map group into the texture database.
		 * @param desc The texture format description, it will be applied to all texture map data in this group.
		 * @return The unique map group ID of the newly inserted texture.
		*/
		[[nodiscard]] STPTextureInformation::STPMapGroupID addMapGroup(const STPMapGroupDescription&);

		/**
		 * @brief Insert a new texture view group into the texture database.
		 * @param desc The texture view description, it will be applied to all texture in this group.
		 * @return The unique view group ID of the newly inserted texture.
		*/
		[[nodiscard]] STPTextureInformation::STPViewGroupID addViewGroup(const STPViewGroupDescription&);

		/**
		 * @brief Remove a texture map group from the texture database, all texture maps in this group will be dropped.
		 * No operation will be performed if map group ID is invalid.
		 * @param group_id The map group with ID to be removed
		*/
		void removeMapGroup(STPTextureInformation::STPMapGroupID);

		/**
		 * @brief Remove a texture view group from the texture database, all texture in this group and all associated texture map will be dropped.
		 * No operation will be performed if view group is invalid.
		 * @param group_id The view group ID to be removed.
		*/
		void removeViewGroup(STPTextureInformation::STPViewGroupID);

		/**
		 * @brief Get the texture map group, given a map group ID.
		 * @param id The texture map group ID.
		 * @return The map group description for requested group.
		 * If group ID is not found, exception is thrown.
		*/
		STPMapGroupDescription getMapGroupDescription(STPTextureInformation::STPMapGroupID) const;

		/**
		 * @brief Get the texture view group, given a view group ID.
		 * @param id The texture view group ID.
		 * @return The view group description for requested group. Exception is thrown if no such group ID is found.
		*/
		STPViewGroupDescription getViewGroupDescription(STPTextureInformation::STPViewGroupID) const;

		/**
		 * @brief Get the number of texture map group registered.
		 * @return The number of registered texture map group.
		*/
		size_t mapGroupSize() const;

		/**
		 * @brief Get the number of texture view group registered.
		 * @return The number of registered texture view group.
		*/
		size_t viewGroupSize() const;

		/**
		 * @brief Insert a new texture into texture database. New texture has no map, and can be added by calling addMap().
		 * A texture may have a collection of different type of maps associated with the texture.
		 * @param group_id The ID of the texture view group. All texture collection in the view group shares the same view properties.
		 * @param name An optional name assigned to this texture for user-identification.
		 * There is no requirement on the format of the name and it is user-defined.
		 * @return The texture ID that can be used to reference the texture
		*/
		[[nodiscard]] STPTextureInformation::STPTextureID addTexture(
			STPTextureInformation::STPViewGroupID, const std::string_view& = std::string_view());

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
		void addMap(STPTextureInformation::STPTextureID, STPTextureType, STPTextureInformation::STPMapGroupID, const void*);

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
#endif//_STP_TEXTURE_DATABASE_H_