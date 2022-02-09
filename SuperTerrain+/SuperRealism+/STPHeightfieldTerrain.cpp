#include <SuperRealism+/Scene/Component/STPHeightfieldTerrain.h>
#include <SuperRealism+/STPRealismInfo.h>
//Noise Generator
#include <SuperRealism+/Utility/STPRandomTextureGenerator.cuh>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
//Indirect
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//GLAD
#include <glad/glad.h>
//CUDA-GL
#include <cuda_gl_interop.h>

#include <array>
#include <limits>

//GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

using std::array;
using std::numeric_limits;
using std::unique_ptr;
using std::make_unique;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::mat4;
using glm::value_ptr;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto HeightfieldTerrainShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPHeightfieldTerrain", 
	".vert", ".tesc", ".tese", ".geom", ".frag");

constexpr static array<unsigned char, 20ull> PlaneVertex = {
	//Position		//Texcoords
	0, 0, 0,		0, 0,
	0, 0, 1,		0, 1,
	1, 0, 1,		1, 1,
	1, 0, 0,		1, 0
};
constexpr static array<unsigned char, 6ull> PlaneIndex = {
	0, 1, 2,
	0, 2, 3
};
constexpr static STPIndirectCommand::STPDrawElement TerrainDrawCommand = {
	static_cast<unsigned int>(PlaneIndex.size()),
	//Needs to set instance count manually during runtime
	1u,
	0u,
	0u,
	0u
};

STPHeightfieldTerrain<false>::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, STPHeightfieldTerrainLog& log, const STPTerrainShaderOption& option) :
	TerrainGenerator(generator_pipeline), NoiseSample(GL_TEXTURE_3D), RandomTextureDimension(option.NoiseDimension) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	const STPDiversity::STPTextureFactory& splatmap_generator = this->TerrainGenerator.splatmapGenerator();
	const STPDiversity::STPTextureInformation::STPSplatTextureDatabase splat_texture = splatmap_generator.getSplatTexture();

	//setup rendering buffer
	this->TileBuffer.bufferStorageSubData(PlaneVertex.data(), PlaneVertex.size() * sizeof(unsigned char), GL_NONE);
	this->TileIndex.bufferStorageSubData(PlaneIndex.data(), PlaneIndex.size() * sizeof(unsigned char), GL_NONE);
	//setup indirect buffer
	const uvec2 tileDimension = chunk_setting.ChunkSize * chunk_setting.RenderedChunk;
	STPIndirectCommand::STPDrawElement cmd = TerrainDrawCommand;
	cmd.InstancedCount = tileDimension.x * tileDimension.y;
	this->TerrainRenderCommand.bufferStorageSubData(&cmd, sizeof(cmd), GL_NONE);
	//attributing
	STPVertexArray::STPVertexAttributeBuilder attr = this->TileArray.attribute();
	attr.format(3, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(unsigned char))
		.format(2, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(unsigned char))
		.vertexBuffer(this->TileBuffer, 0)
		.elementBuffer(this->TileIndex)
		.binding();
	this->TileArray.enable(0u, 2u);

	//setup shader
	STPShaderManager terrain_shader[HeightfieldTerrainShaderFilename.size()] = {
		GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
	};
	for (unsigned int i = 0u; i < HeightfieldTerrainShaderFilename.size(); i++) {
		const char* const source_file = HeightfieldTerrainShaderFilename[i].data();
		STPShaderManager::STPShaderSource shader_source(source_file, *STPFile(source_file));

		if (i == 4u) {
			//fragment shader
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
			//prepare identifiers for texture splatting
			using namespace SuperTerrainPlus::STPDiversity;
			//general info
			Macro("REGION_COUNT", splat_texture.TextureBufferCount)
				("REGISTRY_COUNT", splat_texture.LocationRegistryCount)
				("REGISTRY_DICTIONARY_COUNT", splat_texture.LocationRegistryDictionaryCount)

			//texture type
			("ALBEDO", splatmap_generator.convertType(STPTextureType::Albedo))
			("NORMAL", splatmap_generator.convertType(STPTextureType::Normal))
			("ROUGHNESS", splatmap_generator.convertType(STPTextureType::Roughness))
			("SPECULAR", splatmap_generator.convertType(STPTextureType::Specular))
			("AO", splatmap_generator.convertType(STPTextureType::AmbientOcclusion))

			("TYPE_STRIDE", splatmap_generator.usedType())
			("UNUSED_TYPE", STPTextureFactory::UnusedType)
			("UNREGISTERED_TYPE", STPTextureFactory::UnregisteredType)

			("NORMALMAP_BLENDING", static_cast<std::underlying_type_t<STPNormalBlendingAlgorithm>>(option.NormalBlender));

			//process fragment shader
			shader_source.define(Macro);
		}
		//compile
		log.Log[i] = terrain_shader[i](shader_source);

		//attach the complete terrain program
		if (i < 3u) {
			//anything before geometry shader
			this->TerrainModeller.attach(terrain_shader[i]);
		}
		else {
			//anything after (and include) geometry shader
			this->TerrainShader.attach(terrain_shader[i]);
		}
	}
	this->TerrainModeller.separable(true);
	this->TerrainShader.separable(true);

	//link
	log.Log[5] = this->TerrainModeller.finalise();
	log.Log[6] = this->TerrainShader.finalise();

	//build pipeline
	log.Log[7] = this->TerrainRenderer
		.stage(GL_VERTEX_SHADER_BIT | GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, this->TerrainModeller)
		.stage(GL_GEOMETRY_SHADER_BIT | GL_FRAGMENT_SHADER_BIT, this->TerrainShader)
		.finalise();

	/* ------------------------------- setup initial immutable uniforms ---------------------------------- */
	const vec2 chunkHorizontalOffset = vec2(chunk_setting.ChunkOffset.x, chunk_setting.ChunkOffset.z);
	const vec2 baseChunkPosition = this->calcBaseChunkPosition(chunkHorizontalOffset);
	const uvec2 rendered_chunk = chunk_setting.RenderedChunk;

	//setup program for meshing the terrain
	this->TerrainModeller
		//heightfield for displacement mapping
		.uniform(glProgramUniform1i, "Heightfield", 1)
		//chunk setting
		.uniform(glProgramUniform2uiv, "RenderedChunk", 1, value_ptr(rendered_chunk))
		.uniform(glProgramUniform2uiv, "ChunkSize", 1, value_ptr(chunk_setting.ChunkSize))
		.uniform(glProgramUniform2fv, "BaseChunkPosition", 1, value_ptr(baseChunkPosition));

	//setup program that shades terrain with color
	this->TerrainShader
		//some samplers
		.uniform(glProgramUniform1i, "Biomemap", 0)
		.uniform(glProgramUniform1i, "Heightmap", 1)
		.uniform(glProgramUniform1i, "Splatmap", 2)
		.uniform(glProgramUniform1i, "Noisemap", 3)
		//extra terrain info for rendering
		.uniform(glProgramUniform2uiv, "VisibleChunk", 1, value_ptr(rendered_chunk))
		.uniform(glProgramUniform2fv, "ChunkHorizontalOffset", 1, value_ptr(chunkHorizontalOffset));

	/* --------------------------------- setup texture splatting ------------------------------------ */
	//get splatmap dataset
	const auto& [tbo, tbo_count, reg, reg_count, dict, dict_count] = splat_texture;

	//prepare bindless texture
	this->SplatTextureHandle.reserve(tbo_count);
	unique_ptr<GLuint64[]> rawHandle = make_unique<GLuint64[]>(tbo_count);
	for (unsigned int i = 0u; i < tbo_count; i++) {
		//extract raw handle so we can send them to the shader via uniform
		rawHandle[i] = *this->SplatTextureHandle.emplace_back(tbo[i]);
	}

	//send bindless handle to the shader
	this->TerrainShader.uniform(glProgramUniformHandleui64vARB, "RegionTexture", static_cast<GLsizei>(tbo_count), rawHandle.get())
		//prepare registry
		.uniform(glProgramUniform2uiv, "RegionRegistry", static_cast<GLsizei>(reg_count), reinterpret_cast<const unsigned int*>(reg))
		.uniform(glProgramUniform1uiv, "RegistryDictionary", static_cast<GLsizei>(dict_count), dict);

	/* --------------------------------- shader noise texture preparation ------------------------------ */
	this->NoiseSample.textureStorage<STPTexture::STPDimension::THREE>(1, GL_R8, this->RandomTextureDimension);
	this->NoiseSample.wrap(GL_REPEAT);
	this->NoiseSample.filter(GL_NEAREST, GL_LINEAR);
}

vec2 STPHeightfieldTerrain<false>::calcBaseChunkPosition(const vec2& horizontal_offset) {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->TerrainGenerator.ChunkSetting;
	//calculate offset
	const ivec2 chunk_offset(-glm::floor(vec2(chunk_settings.RenderedChunk) / 2.0f));
	return STPChunk::offsetChunk(horizontal_offset, chunk_settings.ChunkSize, chunk_offset);
}

void STPHeightfieldTerrain<false>::setMesh(const STPEnvironment::STPMeshSetting& mesh_setting) {
	if (!mesh_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Mesh setting is not validated");
	}
	const auto& tess_setting = mesh_setting.TessSetting;
	const auto& smooth_setting = mesh_setting.RegionSmoothSetting;

	//update tessellation LoD control
	this->TerrainModeller.uniform(glProgramUniform1f, "Tess[0].MaxLod", tess_setting.MaxTessLevel)
		.uniform(glProgramUniform1f, "Tess[0].MinLod", tess_setting.MinTessLevel)
		.uniform(glProgramUniform1f, "Tess[0].MaxDis", tess_setting.FurthestTessDistance)
		.uniform(glProgramUniform1f, "Tess[0].MinDis", tess_setting.NearestTessDistance)
		//update other mesh-related parameters
		.uniform(glProgramUniform1f, "Altitude", mesh_setting.Altitude);

	//update settings for rendering
	this->TerrainShader.uniform(glProgramUniform1f, "NormalStrength", mesh_setting.Strength)
		//texture splatting smoothing
		.uniform(glProgramUniform1ui, "SmoothSetting.Kr", smooth_setting.KernelRadius)
		.uniform(glProgramUniform1f, "SmoothSetting.Ks", smooth_setting.KernelScale)
		.uniform(glProgramUniform1f, "SmoothSetting.Ns", smooth_setting.NoiseScale)
		.uniform(glProgramUniform1ui, "UVScaleFactor", mesh_setting.UVScaleFactor);
}

void STPHeightfieldTerrain<false>::seedRandomBuffer(unsigned long long seed) {
	cudaGraphicsResource_t res;
	cudaArray_t random_buffer;

	//register cuda graphics
	STPcudaCheckErr(cudaGraphicsGLRegisterImage(&res, *this->NoiseSample, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsWriteDiscard));
	//map
	STPcudaCheckErr(cudaGraphicsMapResources(1, &res));
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&random_buffer, res, 0u, 0u));

	//compute
	STPCompute::STPRandomTextureGenerator::generate<unsigned char>(random_buffer, this->RandomTextureDimension, seed, 
		numeric_limits<unsigned char>::min(), numeric_limits<unsigned char>::max());

	//clear up
	STPcudaCheckErr(cudaGraphicsUnmapResources(1, &res));
	STPcudaCheckErr(cudaGraphicsUnregisterResource(res));
}

void STPHeightfieldTerrain<false>::setViewPosition(const vec3& viewPos) {
	//prepare heightfield
	if (!this->TerrainGenerator.load(viewPos)) {
		//centre chunk has yet changed, nothing to do.
		return;
	}

	//update model matrix
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	mat4 Model = glm::identity<mat4>();
	//move the terrain centre to the camera
	const vec2& chunkCentre = this->TerrainGenerator.centre();
	Model = glm::translate(Model, vec3(
		chunkCentre.x,
		chunk_setting.ChunkOffset.y,
		chunkCentre.y
	));
	Model = glm::scale(Model, vec3(
		chunk_setting.ChunkScaling,
		1.0f,
		chunk_setting.ChunkScaling
	));
	this->TerrainModeller.uniform(glProgramUniformMatrix4fv, "MeshModel", 1, static_cast<GLboolean>(GL_FALSE), value_ptr(Model));
}

void STPHeightfieldTerrain<false>::render() const {
	//waiting for the heightfield generator to finish
	this->TerrainGenerator.wait();

	//prepare for rendering
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::BIOME]);
	glBindTextureUnit(1, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);
	glBindTextureUnit(2, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::SPLAT]);
	this->NoiseSample.bind(3);

	this->TileArray.bind();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	this->TerrainRenderer.bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPPipelineManager::unbind();
}

STPHeightfieldTerrain<true>::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, STPHeightfieldTerrainLog& log, 
	const STPTerrainShaderOption& option) :
	STPHeightfieldTerrain<false>(generator_pipeline, log, option), MeshQualityLocation(this->TerrainModeller.uniformLocation("ActiveTess")) {

}

void STPHeightfieldTerrain<true>::setDepthMeshQuality(const STPEnvironment::STPMeshSetting::STPTessellationSetting& tess) {
	this->TerrainModeller.uniform(glProgramUniform1f, "Tess[1].MaxLod", tess.MaxTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MinLod", tess.MinTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MaxDis", tess.FurthestTessDistance)
		.uniform(glProgramUniform1f, "Tess[1].MinDis", tess.NearestTessDistance);
}

bool STPHeightfieldTerrain<true>::addDepthConfiguration(size_t light_space_count, const STPShaderManager* depth_shader) {
	STPTerrainDepthLog& log = this->TerrainDepthLogStorage.emplace();
	//create a new render group
	if (this->TerrainDepthRenderer.exist(light_space_count)) {
		//group exists, don't add
		return false;
	}
	auto& [depth_renderer, depth_writer_arr] = this->TerrainDepthRenderer.addGroup(light_space_count);
	auto& [depth_writer] = depth_writer_arr;

	//now the base renderer is finished, setup depth renderer
	STPShaderManager terrain_shadow_shader(GL_GEOMETRY_SHADER);

	//geometry shader for depth writting
	//make a copy of the original source because we need to modify it
	const char* const shadow_source_file = HeightfieldTerrainShaderFilename[3].data();
	STPShaderManager::STPShaderSource shadow_shader_source(shadow_source_file, *STPFile(shadow_source_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("HEIGHTFIELD_SHADOW_PASS", 1)
		("HEIGHTFIELD_SHADOW_PASS_INVOCATION", light_space_count);

	shadow_shader_source.define(Macro);
	log.Log[0] = terrain_shadow_shader(shadow_shader_source);

	//attach program for depth writing
	if (depth_shader) {
		depth_writer.attach(*depth_shader);
	}
	depth_writer.attach(terrain_shadow_shader)
		.separable(true);

	//link
	log.Log[1] = depth_writer.finalise();

	//build shadow pipeline
	log.Log[2] = depth_renderer
		.stage(GL_VERTEX_SHADER_BIT | GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, this->TerrainModeller)
		.stage(GL_GEOMETRY_SHADER_BIT | (depth_shader ? GL_FRAGMENT_SHADER_BIT : 0), depth_writer)
		.finalise();

	return true;
}

void STPHeightfieldTerrain<true>::renderDepth(size_t light_space_count) const {
	this->TerrainGenerator.wait();

	//in this case we only need heightfield for tessellation
	glBindTextureUnit(1, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);

	this->TileArray.bind();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
	//enable low quality mesh
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 1u);

	//find the correct render group
	this->TerrainDepthRenderer.findPipeline(light_space_count).bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPPipelineManager::unbind();
	//change back to normal rendering
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 0u);
}