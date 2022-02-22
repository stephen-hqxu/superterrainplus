#include <SuperRealism+/Geometry/STPPlaneGeometry.h>
#include <SuperRealism+/STPRealismInfo.h>

//GL Object
#include <SuperRealism+/Object/STPBindlessBuffer.h>
#include <SuperRealism+/Object/STPProgramManager.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::uvec2;
using glm::dvec2;
using glm::value_ptr;

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto PlaneGenerationShaderFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPPlaneGeometry", ".comp");

STPPlaneGeometry::STPPlaneGeometry(uvec2 tile_dimension, dvec2 top_left_position, STPPlaneGeometryLog& log) {
	if (tile_dimension.x == 0u || tile_dimension.y == 0u) {
		throw STPException::STPBadNumericRange("Plane geometry must have positive dimension");
	}
	auto& [buffer, index, vertex_array] = this->PlaneData;

	//allocate memory for plane buffer
	const unsigned int tileCount = tile_dimension.x * tile_dimension.y;
	//each tile has two triangle faces, each triangle has 3 vertices
	this->IndexCount = 6ull * tileCount;
	//eacl tile has 4 strides, each stride has a 3-component float as position and 2-component float as texture coordinate
	buffer.bufferStorage(20ull * sizeof(float) * tileCount, GL_NONE);
	index.bufferStorage(sizeof(unsigned int) * this->IndexCount, GL_NONE);
	STPBindlessBuffer buffer_addr(buffer, GL_WRITE_ONLY), 
		index_addr(index, GL_WRITE_ONLY);

	//setup tile buffer generator shader
	STPShaderManager tile_generator(GL_COMPUTE_SHADER);
	const char* const tile_source_file = PlaneGenerationShaderFilename.data();
	STPShaderManager::STPShaderSource tile_source(tile_source_file, *STPFile(tile_source_file));
	//compile shader
	log.Log[0] = tile_generator(tile_source);

	//setup tile generator program
	STPProgramManager plane_generator;
	log.Log[1] = plane_generator
		.attach(tile_generator)
		.finalise();

	//setup up uniform data
	plane_generator.uniform(glProgramUniformui64NV, "TileBuffer", *buffer_addr)
		.uniform(glProgramUniformui64NV, "TileIndex", *index_addr)
		.uniform(glProgramUniform2uiv, "TotalTile", 1, value_ptr(tile_dimension))
		.uniform(glProgramUniform2dv, "BaseTilePosition", 1, value_ptr(top_left_position));

	//vertex array attributing
	STPVertexArray::STPVertexAttributeBuilder attr = vertex_array.attribute();
	attr.format(3, GL_FLOAT, GL_FALSE, sizeof(float))
		.format(2, GL_FLOAT, GL_FALSE, sizeof(float))
		.vertexBuffer(buffer, 0)
		.elementBuffer(index)
		.binding();
	vertex_array.enable(0u, 2u);

	/* ------------------------------------ prepare for generation ------------------------------------- */
	//calculate generation size
	const uvec2 dimBlockSize = static_cast<uvec2>(plane_generator.workgroupSize()),
		dimGridSize = (tile_dimension + dimBlockSize - 1u) / dimBlockSize;

	//dispatch
	plane_generator.use();
	glDispatchCompute(dimGridSize.x, dimGridSize.y, 1u);
	//make sure all buffer written are visible for rests of the rendering commands
	glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV);

	//clear up
	STPProgramManager::unuse();
}

const STPPlaneGeometry::STPPlaneGeometryData& STPPlaneGeometry::operator*() const {
	return this->PlaneData;
}

unsigned int STPPlaneGeometry::planeIndexCount() const {
	return this->IndexCount;
}

void STPPlaneGeometry::bindPlaneVertexArray() const {
	this->PlaneData.PlaneArray.bind();
}