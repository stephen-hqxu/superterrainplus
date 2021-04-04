#pragma once
#include "STPProcedural2DINF.h"

using std::cout;
using std::cerr;
using std::endl;

using glm::vec2;
using glm::mat4;
using glm::identity;

using namespace SuperTerrainPlus;

STPProcedural2DINF::STPProcedural2DINF(STPSettings::STPConfigurations* const settings, void* const procedural2dinf_cmd)
	: STPChunkManager(settings), command(procedural2dinf_cmd), RenderingSettings(settings->getMeshSettings()) {
	cout << "....Loading STPProcedural2DINF, An Infinite Terrain Renderer....";
	if (this->compile2DTerrainShader()) {
		cout << "Shader Loaded :)" << endl;
	}
	this->calcBaseChunkPosition();
	this->loadPlane();
	cout << "....Done...." << endl;
}

STPProcedural2DINF::~STPProcedural2DINF() {
	this->clearup();
}

const bool STPProcedural2DINF::compile2DTerrainShader() {
	const STPSettings::STPChunkSettings* chunk_settings = this->getChunkProvider().getChunkSettings();
	//log
	GLchar* log = new GLchar[1024];
	
	this->Terrain2d_shader.addShader(GL_VERTEX_SHADER, "./GLSL/STPProcedural2DINF.vert");
	this->Terrain2d_shader.addShader(GL_TESS_CONTROL_SHADER, "./GLSL/STPProcedural2DINF.tesc");
	this->Terrain2d_shader.addShader(GL_TESS_EVALUATION_SHADER, "./GLSL/STPProcedural2DINF.tese");
	this->Terrain2d_shader.addShader(GL_GEOMETRY_SHADER, "./GLSL/STPProcedural2DINF.geom");
	this->Terrain2d_shader.addShader(GL_FRAGMENT_SHADER, "./GLSL/STPProcedural2DINF.frag");
	if (this->Terrain2d_shader.linkShader(log, 1024, [](GLuint program) -> void {
		glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
		}) != SglToolkit::SgTShaderProc::OK) {

		cerr << "\n----------------STPProcedural2DINF doesn't seem to work :(-------------------" << endl;
		cerr << log << endl;
		cerr << "-------------------------------------------------------------------------------" << endl;
		//exit
		delete[] log;
		return false;
	}
	//binding sampler
	glProgramUniform1i(this->Terrain2d_shader.getP(), this->getLoc("Heightmap"), 0);
	glProgramUniform1i(this->Terrain2d_shader.getP(), this->getLoc("Normalmap_Terrain"), 1);
	//those parameters won't change, there is no need to resend them in rendering loop
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("rendered_chunk_num"), 1, value_ptr(chunk_settings->RenderedChunk));
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("chunk_dimension"), 1, value_ptr(chunk_settings->ChunkSize));
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.MAX_TESS_LEVEL"), this->RenderingSettings.TessSettings.MaxTessLevel);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.MIN_TESS_LEVEL"), this->RenderingSettings.TessSettings.MinTessLevel);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.FURTHEST_TESS_DISTANCE"), this->RenderingSettings.TessSettings.FurthestTessDistance);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.NEAREST_TESS_DISTANCE"), this->RenderingSettings.TessSettings.NearestTessDistance);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("altitude"), this->RenderingSettings.Altitude);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("shiftFactor"), this->RenderingSettings.LoDShiftFactor);

	//create pipeline
	glCreateProgramPipelines(1, &this->Terrain2d_pipeline);
	glUseProgramStages(this->Terrain2d_pipeline, GL_ALL_SHADER_BITS, this->Terrain2d_shader.getP());
	
	//clerup
	delete[] log;
	return true;
}

void STPProcedural2DINF::calcBaseChunkPosition() {
	const STPSettings::STPChunkSettings* chunk_settings = this->getChunkProvider().getChunkSettings();
	//allocate space
	const unsigned int num_chunk = chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y;
	this->BaseChunkPosition = new vec2[num_chunk];

	const auto rendered_chunks = STPChunk::getRegion(vec2(chunk_settings->ChunkOffset.x, chunk_settings->ChunkOffset.z),
		chunk_settings->ChunkSize, chunk_settings->RenderedChunk);
	std::copy(rendered_chunks.begin(), rendered_chunks.end(), this->BaseChunkPosition);
}

void STPProcedural2DINF::loadPlane() {
	const STPSettings::STPChunkSettings* chunk_settings = this->getChunkProvider().getChunkSettings();
	//create buffers
	glCreateBuffers(1, &this->plane_vbo);
	glCreateBuffers(1, &this->plane_indirect);
	glCreateVertexArrays(1, &this->plane_vao);
	glCreateBuffers(1, &this->plane_ebo);
	glCreateBuffers(1, &this->plane_basechunkposition);

	//create indirect buffer
	//the size is just sizeof(DrawElementsIndirectCommand), that type contains 5 unsigned int
	//since that is not part of the terrain engine, I am not going to include that to this file
	glNamedBufferStorage(this->plane_indirect, sizeof(GLuint) * 5, this->command, GL_NONE);

	//sending data
	glNamedBufferStorage(this->plane_vbo, sizeof(SglToolkit::SgTUtils::UNITPLANE_VERTICES), SglToolkit::SgTUtils::UNITPLANE_VERTICES, GL_NONE);
	glNamedBufferStorage(this->plane_ebo, sizeof(SglToolkit::SgTUtils::UNITBOX_INDICES), SglToolkit::SgTUtils::UNITBOX_INDICES, GL_NONE);
	glNamedBufferStorage(this->plane_basechunkposition, 
		sizeof(vec2) * chunk_settings->RenderedChunk.x * chunk_settings->RenderedChunk.y, this->BaseChunkPosition, GL_NONE);
	//binding to vao
	glVertexArrayVertexBuffer(this->plane_vao, 0, this->plane_vbo, 0, 14 * sizeof(int));
	glVertexArrayElementBuffer(this->plane_vao, this->plane_ebo);
	glVertexArrayVertexBuffer(this->plane_vao, 1, this->plane_basechunkposition, 0, sizeof(vec2));
	//attributing
	for (int i = 0; i < 6; i++) {
		glEnableVertexArrayAttrib(this->plane_vao, i);
	}
	//Binding block 0
	glVertexArrayAttribFormat(this->plane_vao, 0, 3, GL_INT, GL_FALSE, 0);//vertex
	glVertexArrayAttribFormat(this->plane_vao, 1, 2, GL_INT, GL_FALSE, sizeof(int) * 3);//uv
	glVertexArrayAttribFormat(this->plane_vao, 2, 3, GL_INT, GL_FALSE, sizeof(int) * 5);//normal
	glVertexArrayAttribFormat(this->plane_vao, 3, 3, GL_INT, GL_FALSE, sizeof(int) * 8);//tangent
	glVertexArrayAttribFormat(this->plane_vao, 4, 3, GL_INT, GL_FALSE, sizeof(int) * 11);//bitangent
	//Binding block 1
	glVertexArrayAttribFormat(this->plane_vao, 5, 2, GL_FLOAT, GL_FALSE, 0);//base chunk position
	//Block 0
	for (int i = 0; i < 5; i++) {
		glVertexArrayAttribBinding(this->plane_vao, i, 0);
	}
	//Block 1
	glVertexArrayAttribBinding(this->plane_vao, 5, 1);
	//we send new chunk data per chunk.
	//our rendering call is based on unit plane so we update the attribute every chunk_size.x*y call
	glVertexArrayBindingDivisor(this->plane_vao, 1, chunk_settings->ChunkSize.x * chunk_settings->ChunkSize.y);
	
	return;
}

GLint STPProcedural2DINF::getLoc(const GLchar* const name) const {
	return glGetUniformLocation(this->Terrain2d_shader.getP(), name);
}

void STPProcedural2DINF::clearup() {
	//clearup
	delete[] this->BaseChunkPosition;
	this->Terrain2d_shader.deleteShader();
	glDeleteBuffers(1, &this->plane_vbo);
	glDeleteBuffers(1, &this->plane_indirect);
	glDeleteVertexArrays(1, &this->plane_vao);
	glDeleteBuffers(1, &this->plane_ebo);
	glDeleteBuffers(1, &this->plane_basechunkposition);
	glDeleteProgramPipelines(1, &this->Terrain2d_pipeline);
}

GLuint STPProcedural2DINF::getTerrain2DINFProgram() {
	return this->Terrain2d_shader.getP();
}

void STPProcedural2DINF::renderVisibleChunks(const mat4& view, const mat4& projection, const vec3& position) {
	const STPSettings::STPChunkSettings* chunk_settings = this->getChunkProvider().getChunkSettings();
	mat4 model = identity<mat4>();
	//move the terrain center to the camera
	vec2 chunk_center_position = STPChunk::getChunkPosition(
		position - chunk_settings->ChunkOffset, chunk_settings->ChunkSize, chunk_settings->ChunkScaling);
	model = glm::translate(model, 
		vec3(chunk_center_position.x, chunk_settings->ChunkOffset.y, chunk_center_position.y));
	model = glm::scale(model, 
		vec3(chunk_settings->ChunkScaling, 1.0f, chunk_settings->ChunkScaling));

	//sending informations for every loop, uniforms that remain constant have been sent
	glProgramUniform3fv(this->Terrain2d_shader.getP(), this->getLoc("cameraPos"), 1, value_ptr(position));
	glProgramUniformMatrix4fv(this->Terrain2d_shader.getP(), this->getLoc("Model"), 1, GL_FALSE, value_ptr(model));
	
	//render, sync textures to make sure they are all ready before loading
	this->SyncloadChunks();
	this->generateMipmaps();
	glBindTextureUnit(0, *(this->terrain_heightfield));//heightmap
	glBindTextureUnit(1, *(this->terrain_heightfield + 1));//normalmap
	//terrain surface texture isn't ready yet
	//const unsigned int instance_count = this->CHUNK_SIZE.x * this->CHUNK_SIZE.y * this->RENDERED_CHUNK.x * this->RENDERED_CHUNK.y;
	glBindVertexArray(this->plane_vao);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, this->plane_indirect);
	glBindProgramPipeline(this->Terrain2d_pipeline);
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);
	glBindProgramPipeline(0);
}