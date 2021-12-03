#include "STPProcedural2DINF.h"

using std::cout;
using std::cerr;
using std::endl;

using std::make_unique;

#include <glm/gtc/type_ptr.hpp>

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;
using glm::mat4;
using glm::identity;

using namespace STPDemo;
using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPDiversity;

STPProcedural2DINF::STPProcedural2DINF(const STPEnvironment::STPMeshSetting& mesh_settings, STPWorldPipeline& pipeline, void* procedural2dinf_cmd)
	: WorldPipeline(pipeline), command(procedural2dinf_cmd), MeshSetting(mesh_settings) {
	cout << "....Loading STPProcedural2DINF, An Infinite Terrain Renderer....";
	this->compile2DTerrainShader();
	cout << "Shader Loaded :)" << endl;
	this->loadPlane();
	this->loadTexture();
	cout << "....Done...." << endl;
}

STPProcedural2DINF::~STPProcedural2DINF() {
	this->clearup();
}

void STPProcedural2DINF::compile2DTerrainShader() {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldPipeline.ChunkSetting;
	//log
	GLchar log[1024];
	
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
		std::terminate();
	}
	//binding sampler
	glProgramUniform1i(this->Terrain2d_shader.getP(), this->getLoc("Biomemap"), 0);
	glProgramUniform1i(this->Terrain2d_shader.getP(), this->getLoc("Heightfield"), 1);
	glProgramUniform1i(this->Terrain2d_shader.getP(), this->getLoc("Splatmap"), 2);
	//those parameters won't change, there is no need to resend them in rendering loop
	const vec2 base_chunk_position = this->calcBaseChunkPosition();
	const uvec2 rendering_buffer_size = chunk_settings.RenderedChunk * chunk_settings.MapSize;
	const vec2 chunk_horizontal_offset = vec2(chunk_settings.ChunkOffset.x, chunk_settings.ChunkOffset.z);
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("rendered_chunk_num"), 1, value_ptr(chunk_settings.RenderedChunk));
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("chunk_dimension"), 1, value_ptr(chunk_settings.ChunkSize));
	glProgramUniform2fv(this->Terrain2d_shader.getP(), this->getLoc("base_chunk_position"), 1, value_ptr(base_chunk_position));
	glProgramUniform2fv(this->Terrain2d_shader.getP(), this->getLoc("chunk_offset"), 1, value_ptr(chunk_horizontal_offset));
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.MAX_TESS_LEVEL"), this->MeshSetting.TessSetting.MaxTessLevel);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.MIN_TESS_LEVEL"), this->MeshSetting.TessSetting.MinTessLevel);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.FURTHEST_TESS_DISTANCE"), this->MeshSetting.TessSetting.FurthestTessDistance);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("tessParameters.NEAREST_TESS_DISTANCE"), this->MeshSetting.TessSetting.NearestTessDistance);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("altitude"), this->MeshSetting.Altitude);
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("shiftFactor"), this->MeshSetting.LoDShiftFactor);
	//Geometry shader for normalmap calculation
	glProgramUniform1f(this->Terrain2d_shader.getP(), this->getLoc("NormalStrength"), this->MeshSetting.Strength);
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("HeightfieldDim"), 1, value_ptr(rendering_buffer_size));
	//Fragment shader for texture splatting
	const STPTextureFactory& splatGen = this->WorldPipeline.splatmapGenerator();
	//texture type indexer
	glProgramUniform1ui(this->Terrain2d_shader.getP(), this->getLoc("ALBEDO"), splatGen.convertType(STPTextureType::Albedo));

	//create pipeline
	glCreateProgramPipelines(1, &this->Terrain2d_pipeline);
	glUseProgramStages(this->Terrain2d_pipeline, GL_ALL_SHADER_BITS, this->Terrain2d_shader.getP());
}

vec2 STPProcedural2DINF::calcBaseChunkPosition() {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldPipeline.ChunkSetting;
	//calculate offset
	const ivec2 chunk_offset(-glm::floor(vec2(chunk_settings.RenderedChunk) / 2.0f));
	return STPChunk::offsetChunk(vec2(chunk_settings.ChunkOffset.x, chunk_settings.ChunkOffset.z), chunk_settings.ChunkSize, chunk_offset);
}

void STPProcedural2DINF::loadPlane() {
	//create buffers
	glCreateBuffers(1, &this->plane_vbo);
	glCreateBuffers(1, &this->plane_indirect);
	glCreateVertexArrays(1, &this->plane_vao);
	glCreateBuffers(1, &this->plane_ebo);

	//create indirect buffer
	//the size is just sizeof(DrawElementsIndirectCommand), that type contains 5 unsigned int
	//since that is not part of the terrain engine, I am not going to include that to this file
	glNamedBufferStorage(this->plane_indirect, sizeof(GLuint) * 5, this->command, GL_NONE);

	//sending data
	glNamedBufferStorage(this->plane_vbo, sizeof(SglToolkit::SgTUtil::UNITPLANE_VERTICES), SglToolkit::SgTUtil::UNITPLANE_VERTICES, GL_NONE);
	glNamedBufferStorage(this->plane_ebo, sizeof(SglToolkit::SgTUtil::UNITBOX_INDICES), SglToolkit::SgTUtil::UNITBOX_INDICES, GL_NONE);
	//binding to vao
	glVertexArrayVertexBuffer(this->plane_vao, 0, this->plane_vbo, 0, 14 * sizeof(int));
	glVertexArrayElementBuffer(this->plane_vao, this->plane_ebo);
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
	//Block 0 binding
	for (int i = 0; i < 5; i++) {
		glVertexArrayAttribBinding(this->plane_vao, i, 0);
	}
	
	return;
}

void STPProcedural2DINF::loadTexture() {
	const STPTextureFactory& splatGen = this->WorldPipeline.splatmapGenerator();
	//get all splatmap dataset
	const auto [tbo, tbo_count, reg, reg_count, dict, dict_count] = splatGen.getSplatTexture();
	
	//prepare bindless texture
	this->SplatTextureHandle.reserve(tbo_count);
	for (unsigned int i = 0u; i < tbo_count; i++) {
		const GLuint64 handle = glGetTextureHandleARB(tbo[i]);
		this->SplatTextureHandle.emplace_back(handle);
		//make the handle active
		glMakeTextureHandleResidentARB(handle);
	}
	glProgramUniformHandleui64vARB(this->Terrain2d_shader.getP(), this->getLoc("RegionalTexture"), tbo_count, this->SplatTextureHandle.data());

	//prepare registry
	glProgramUniform2uiv(this->Terrain2d_shader.getP(), this->getLoc("RegionLocation"), reg_count, reinterpret_cast<const unsigned int*>(reg));
	//prepare dictionary for registry
	glProgramUniform1uiv(this->Terrain2d_shader.getP(), this->getLoc("RegionLocationDictionary"), dict_count, dict);
}

GLint STPProcedural2DINF::getLoc(const GLchar* const name) const {
	return glGetUniformLocation(this->Terrain2d_shader.getP(), name);
}

void STPProcedural2DINF::clearup() {
	//free bindless handles
	for (const auto handle : this->SplatTextureHandle) {
		glMakeTextureHandleNonResidentARB(handle);
	}

	//clearup
	this->Terrain2d_shader.deleteShader();
	glDeleteBuffers(1, &this->plane_vbo);
	glDeleteBuffers(1, &this->plane_indirect);
	glDeleteVertexArrays(1, &this->plane_vao);
	glDeleteBuffers(1, &this->plane_ebo);
	glDeleteProgramPipelines(1, &this->Terrain2d_pipeline);
}

GLuint STPProcedural2DINF::getTerrain2DINFProgram() const {
	return this->Terrain2d_shader.getP();
}

void STPProcedural2DINF::renderVisibleChunks(const mat4& view, const mat4& projection, vec3 position) const {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->WorldPipeline.ChunkSetting;
	mat4 model = identity<mat4>();
	//move the terrain center to the camera
	vec2 chunk_center_position = STPChunk::getChunkPosition(
		position - chunk_settings.ChunkOffset, chunk_settings.ChunkSize, chunk_settings.ChunkScaling);
	model = glm::translate(model, 
		vec3(chunk_center_position.x, chunk_settings.ChunkOffset.y, chunk_center_position.y));
	model = glm::scale(model, 
		vec3(chunk_settings.ChunkScaling, 1.0f, chunk_settings.ChunkScaling));

	//sending informations for every loop, uniforms that remain constant have been sent
	glProgramUniform3fv(this->Terrain2d_shader.getP(), this->getLoc("cameraPos"), 1, value_ptr(position));
	glProgramUniformMatrix4fv(this->Terrain2d_shader.getP(), this->getLoc("Model"), 1, GL_FALSE, value_ptr(model));
	
	//render, sync textures to make sure they are all ready before loading
	try {
		this->WorldPipeline.wait();
	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
		std::terminate();
	}

	glBindTextureUnit(0, this->WorldPipeline[STPWorldPipeline::STPRenderingBufferType::BIOME]);//biomemap
	glBindTextureUnit(1, this->WorldPipeline[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);//heightfield
	glBindTextureUnit(2, this->WorldPipeline[STPWorldPipeline::STPRenderingBufferType::SPLAT]);//splatmap
	//terrain surface texture isn't ready yet
	//const unsigned int instance_count = this->CHUNK_SIZE.x * this->CHUNK_SIZE.y * this->RENDERED_CHUNK.x * this->RENDERED_CHUNK.y;
	glBindVertexArray(this->plane_vao);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, this->plane_indirect);
	glBindProgramPipeline(this->Terrain2d_pipeline);
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);
	glBindProgramPipeline(0);
}