#include <SuperRealism+/Geometry/STPPlaneGeometry.h>
#include <SuperRealism+/STPRealismInfo.h>

//GL Object
#include <SuperRealism+/Object/STPProgramManager.h>

//Error
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

//GLAD
#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

using glm::uvec2;
using glm::dvec2;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto PlaneGenerationShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPPlaneGeometry", ".comp");

STPPlaneGeometry::STPPlaneGeometry(const uvec2 tile_dimension, const dvec2 top_left_position) {
	STP_ASSERTION_NUMERIC_DOMAIN(tile_dimension.x > 0u && tile_dimension.y > 0u, "Plane geometry must have positive dimension");
	auto& [buffer, index, vertex_array] = this->PlaneData;

	//allocate memory for plane buffer
	//see documentation in the shader to understand how memory usage is calculated
	const uvec2 vertex_dimension = tile_dimension + 1u;
	const unsigned int vertexCount = vertex_dimension.x * vertex_dimension.y,
		tileCount = tile_dimension.x * tile_dimension.y;
	//each tile has two triangle faces, each triangle has 3 vertices
	this->IndexCount = 6u * tileCount;
	//each vertex has a 2-component float as position and 2-component float as texture coordinate
	buffer.bufferStorage(4u * sizeof(float) * vertexCount, GL_NONE);
	index.bufferStorage(sizeof(unsigned int) * this->IndexCount, GL_NONE);
	const GLuint64EXT buffer_addr = buffer.getAddress(),
		index_addr = index.getAddress();
	buffer.makeResident(GL_WRITE_ONLY);
	index.makeResident(GL_WRITE_ONLY);

	//setup tile buffer generator shader
	const char* const tile_source_file = PlaneGenerationShaderFilename.data();
	STPShaderManager::STPShaderSource tile_source(tile_source_file, STPFile::read(tile_source_file));
	//compile shader
	const STPShaderManager::STPShader tile_generator = STPShaderManager::make(GL_COMPUTE_SHADER, tile_source);

	//setup tile generator program
	STPProgramManager plane_generator({ &tile_generator });

	//setup up uniform data
	plane_generator.uniform(glProgramUniformui64NV, "TileBuffer", buffer_addr)
		.uniform(glProgramUniformui64NV, "TileIndex", index_addr)
		.uniform(glProgramUniform2uiv, "TotalTile", 1, value_ptr(tile_dimension))
		.uniform(glProgramUniform2dv, "BaseTilePosition", 1, value_ptr(top_left_position));

	//vertex array attributing
	STPVertexArray::STPVertexAttributeBuilder attr = vertex_array.attribute();
	attr.format(2, GL_FLOAT, GL_FALSE, sizeof(float))
		.format(2, GL_FLOAT, GL_FALSE, sizeof(float))
		.vertexBuffer(buffer, 0)
		.elementBuffer(index)
		.binding();
	vertex_array.enable(0u, 2u);

	//prepare for plane geometry generation
	auto launchKernel = [dimBlockSize = static_cast<uvec2>(plane_generator.workgroupSize())](uvec2 count) -> void {
		const uvec2 dimGridSize = (count + dimBlockSize - 1u) / dimBlockSize;
		glDispatchCompute(dimGridSize.x, dimGridSize.y, 1u);
	};
	const STPProgramManager::STPProgramStateManager plane_gen_state = plane_generator.useManaged();
	GLuint generation_pass;
	/* ---------------------------- plane vertex generation ------------------------ */
	generation_pass = 0u;
	glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1u, &generation_pass);

	launchKernel(vertex_dimension);
	/* ---------------------------- plane index generation ------------------------- */
	generation_pass = 1u;
	glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1u, &generation_pass);

	launchKernel(uvec2(tile_dimension.x * 2u, tile_dimension.y));

	//make sure all buffer written are visible for rests of the rendering commands
	glMemoryBarrier(GL_SHADER_GLOBAL_ACCESS_BARRIER_BIT_NV);

	//clear up
	buffer.makeNonResident();
	index.makeNonResident();
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