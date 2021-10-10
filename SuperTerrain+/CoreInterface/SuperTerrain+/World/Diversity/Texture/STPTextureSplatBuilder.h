#pragma once
#ifndef _STP_TEXTURE_SPLAT_BUILDER_H_
#define _STP_TEXTURE_SPLAT_BUILDER_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Biome
#include "../STPBiomeDefine.h"
#include "STPTextureInformation.hpp"

//Container
#include <list>
#include <map>
#include <vector>

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
		 * @brief STPTextureSplatBuilder is a simple utility that allows each biome to have different texture.
		 * Texture with a biome can be arranged either by altitude or gradient at any point on the terrain mesh.
		*/
		class STP_API STPTextureSplatBuilder {
		private:

			//an array of non-owning biome structure
			template<class S>
			using STPStructureView = std::vector<std::pair<Sample, const S*>>;

		public:

			//A structure to define layers of texture splating using altitude rule
			//Upper bound of the altitude is used as the key, the texture ID defines the texture used in the region
			typedef std::map<float, STPTextureInformation::STPAltitudeNode> STPAltitudeStructure;
			//A structure to define layers of texture splatting using gradient rule
			typedef std::list<STPTextureInformation::STPGradientNode> STPGradientStructure;

			//array of non-owning biome altitude structure
			typedef STPStructureView<STPAltitudeStructure> STPAltitudeView;
			//array of non-owning biome gradient structure
			typedef STPStructureView<STPGradientStructure> STPGradientView;

		private:

			std::unordered_map<Sample, STPAltitudeStructure> BiomeAltitudeMapping;
			std::unordered_map<Sample, STPGradientStructure> BiomeGradientMapping;

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
			 * @brief Init STPTextureSplatBuilder as default
			*/
			STPTextureSplatBuilder() = default;

			STPTextureSplatBuilder(const STPTextureSplatBuilder&) = delete;

			STPTextureSplatBuilder(STPTextureSplatBuilder&&) noexcept = default;

			STPTextureSplatBuilder& operator=(const STPTextureSplatBuilder&) = delete;

			STPTextureSplatBuilder& operator=(STPTextureSplatBuilder&&) noexcept = default;

			~STPTextureSplatBuilder() = default;

			/**
			 * @brief Get pointer to altitude structure for the specified sample
			 * @param sample The altitude for the sample to be retrieved
			 * @return The pointer to the altitude structure for the sample.
			 * If no altitude is associated with said sample, exception is thrown
			*/
			const STPAltitudeStructure& getAltitude(Sample) const;

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
			const STPGradientStructure& getGradient(Sample) const;

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
			void addAltitude(Sample, float, STPTextureDatabase::STPTextureID);

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
			void addGradient(Sample, float, float, float, float, STPTextureDatabase::STPTextureID);

			/**
			 * @brief Add a set of new configurations for specified biome into gradient structure
			 * @tparam ...Arg Argument pack to be added. Must follow the argument structure of addGradient() function
			 * @param sample The sample that the new gradient configurations belong to
			 * @param ...args Arguments grouped to be addded
			*/
			template<class... Arg>
			void addGradients(Sample, Arg&&...);

		};

	}
}
#include "STPTextureSplatBuilder.inl"
#endif//_STP_TEXTURE_SPLAT_BUILDER_H_