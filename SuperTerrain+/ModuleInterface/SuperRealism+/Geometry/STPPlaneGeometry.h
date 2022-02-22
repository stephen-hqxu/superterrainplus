#pragma once
#ifndef _STP_PLANE_GEOMETRY_H_
#define _STP_PLANE_GEOMETRY_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPBuffer.h"
#include "../Object/STPVertexArray.h"

#include "../Utility/STPLogStorage.hpp"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPPlaneGeometry is a geometry of flat plane with no thickness.
	 * One plane contains 2 triangular faces, each vertex has a 3-component position and 2-component texture coordinate.
	 * Like most geometry, they are specified as single-precision floating point.
	*/
	class STP_REALISM_API STPPlaneGeometry {
	public:

		/**
		 * @brief STPPlaneGeometryData stores data for a plane geometry.
		*/
		struct STPPlaneGeometryData {
		public:

			STPBuffer PlaneBuffer, PlaneIndex;
			STPVertexArray PlaneArray;

		};

	private:

		//Generated plane data, this data will be in model space.
		STPPlaneGeometryData PlaneData;

		unsigned int IndexCount;

	public:

		typedef STPLogStorage<2ull> STPPlaneGeometryLog;

		/**
		 * @brief Initialise a new plane geometry and generate a new plane geometry.
		 * @param tile_dimension The number of tile on the plane in x and y direction.
		 * The plane will be subdivided into equal sized tiles.
		 * @param top_left_position Specifies top-left position of the plane geometry model.
		 * @param log The log to be returned for compilation of plane generator shader.
		*/
		STPPlaneGeometry(glm::uvec2, glm::dvec2, STPPlaneGeometryLog&);

		STPPlaneGeometry(const STPPlaneGeometry&) = delete;

		STPPlaneGeometry(STPPlaneGeometry&&) = delete;

		STPPlaneGeometry& operator=(const STPPlaneGeometry&) = delete;

		STPPlaneGeometry& operator=(STPPlaneGeometry&&) = delete;

		~STPPlaneGeometry() = default;

		/**
		 * @brief Get the underlying plane geometry.
		 * @return The pointer to the plane geometry data.
		*/
		const STPPlaneGeometryData& operator*() const;

		/**
		 * @brief Get the number of index for the plane geometry.
		 * @return The number of index.
		*/
		unsigned int planeIndexCount() const;

		/**
		 * @brief Bind the vertex array object for the plane geometry.
		*/
		void bindPlaneVertexArray() const;

	};

}
#endif//_STP_PLANE_GEOMETRY_H_