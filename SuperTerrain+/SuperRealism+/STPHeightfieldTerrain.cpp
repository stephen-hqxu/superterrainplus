#include <SuperRealism+/Renderer/STPHeightfieldTerrain.h>
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
//Indirect
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//GLAD
#include <glad/glad.h>

//Container
#include <array>

//GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

using std::array;
using std::string;
using std::to_string;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::vec3;
using glm::mat4;
using glm::value_ptr;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto HeightfieldTerrainShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPHeightfieldTerrain", 
	".vert", ".tesc", ".tese", ".geom", ".frag");

constexpr static array<signed char, 56ull> PlaneVertex = {
	//Position		//Texcoords		//Normal	//Tangent	//Bitangent
	0, 0, 0,		0, 0,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	1, 0, 0,		1, 0,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	1, 0, 1,		1, 1,			0, 1, 0,	1, 0, 0,	0, 0, -1,
	0, 0, 1,		0, 1,			0, 1, 0,	1, 0, 0,	0, 0, -1
};
constexpr static array<unsigned char, 6ull> PlaneIndex = {
	0, 1, 2,
	0, 2, 3
};
constexpr static STPIndirectCommand::STPDrawElement TerrainDrawCommand = {
	PlaneIndex.size(),
	//Needs to set instance count manually during runtime
	1u,
	0u,
	0u,
	0u
};

STPHeightfieldTerrain::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, STPHeightfieldTerrainLog& log) : 
	TerrainGenerator(generator_pipeline), ViewPosition(vec3(0.0f)) {
	if (!GLAD_GL_ARB_bindless_texture) {
		throw STPException::STPUnsupportedFunctionality("The current rendering context does not support ARB_bindless_texture");
	}

	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	const STPDiversity::STPTextureFactory& splatmap_generator = this->TerrainGenerator.splatmapGenerator();
	const STPDiversity::STPTextureInformation::STPSplatTextureDatabase splat_texture = splatmap_generator.getSplatTexture();

	//setup rendering buffer
	this->TileBuffer.bufferStorageSubData(PlaneVertex.data(), PlaneVertex.size() * sizeof(signed char), GL_NONE);
	this->TileIndex.bufferStorageSubData(PlaneIndex.data(), PlaneIndex.size() * sizeof(unsigned char), GL_NONE);
	//setup indirect buffer
	const uvec2 tileDimension = chunk_setting.ChunkSize * chunk_setting.RenderedChunk;
	STPIndirectCommand::STPDrawElement cmd = TerrainDrawCommand;
	cmd.InstancedCount = tileDimension.x * tileDimension.y;
	this->TerrainRenderCommand.bufferStorageSubData(&cmd, sizeof(cmd), GL_NONE);
	//attributing
	STPVertexArray::STPVertexAttributeBuilder attr = this->TileArray.attribute();
	attr.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->TileBuffer, 0)
		.elementBuffer(this->TileIndex)
		.binding();
	this->TileArray.enable(0u, 5u);

	//setup shader
	STPShaderManager terrain_shader[HeightfieldTerrainShaderFilename.size()] = {
		GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
	};
	for (unsigned int i = 0u; i < HeightfieldTerrainShaderFilename.size(); i++) {
		STPShaderManager& current_shader = terrain_shader[i];
		const char* const terrain_filename = HeightfieldTerrainShaderFilename[i].data();

		const STPFile shader_source(terrain_filename);
		//compile
		if (i == 4u) {
			//prepare compile-time macros
			STPShaderManager::STPMacroValueDictionary Macro;
			//prepare identifiers for texture splatting
			using namespace SuperTerrainPlus::STPDiversity;
			//general info
			Macro["REGION_COUNT"] = to_string(splat_texture.TextureBufferCount);
			Macro["REGISTRY_COUNT"] = to_string(splat_texture.LocationRegistryCount);
			Macro["REGISTRY_DICTIONARY_COUNT"] = to_string(splat_texture.LocationRegistryDictionaryCount);

			//texture type
			Macro["ALBEDO"] = to_string(splatmap_generator.convertType(STPTextureType::Albedo));
			Macro["UNUSED_TYPE"] = to_string(STPTextureFactory::UnusedType);
			Macro["UNREGISTERED_TYPE"] = to_string(STPTextureFactory::UnregisteredType);

			//process fragment shader
			current_shader.cache(*shader_source);
			current_shader.defineMacro(Macro);
			log.Log[i] = current_shader();
		}
		else {
			//add include path for tes control and geometry shader
			log.Log[i] = (i == 1u || i == 3u) ? current_shader(*shader_source, { "/Common/STPCameraInformation.glsl" }) : current_shader(*shader_source);
		}

		//attach to program
		this->TerrainComponent.attach(current_shader);
	}
	this->TerrainComponent.setSeparable(true);
	//link
	this->TerrainComponent.finalise();
	log.Log[5] = this->TerrainComponent.lastLog();
	if (!this->TerrainComponent) {
		//program not usable
		throw STPException::STPGLError("Heightfield terrain renderer program returns a failed status");
	}

	//build pipeline
	log.Log[6] = this->TerrainRenderer.stage(GL_ALL_SHADER_BITS, this->TerrainComponent)
		.finalise();

	/* ------------------------------- setup initial immutable uniforms ---------------------------------- */
	const vec2 chunkHorizontalOffset = vec2(chunk_setting.ChunkOffset.x, chunk_setting.ChunkOffset.z);
	const uvec2 textureResolution = chunk_setting.RenderedChunk * chunk_setting.MapSize;
	const vec2 baseChunkPosition = this->calcBaseChunkPosition(chunkHorizontalOffset);
	//assign sampler
	this->TerrainComponent.uniform(glProgramUniform1i, "Biomemap", 0)
		.uniform(glProgramUniform1i, "Heightfield", 1)
		.uniform(glProgramUniform1i, "Splatmap", 2)
		//chunk setting
		.uniform(glProgramUniform2uiv, "ChunkSize", 1, value_ptr(chunk_setting.ChunkSize))
		.uniform(glProgramUniform2uiv, "RenderedChunk", 1, value_ptr(chunk_setting.RenderedChunk))
		.uniform(glProgramUniform2fv, "ChunkOffset", 1, value_ptr(chunkHorizontalOffset))
		.uniform(glProgramUniform2uiv, "HeightfieldResolution", 1, value_ptr(textureResolution))
		.uniform(glProgramUniform2fv, "BaseChunkPosition", 1, value_ptr(baseChunkPosition));

	/* --------------------------------- setup texture splatting ------------------------------------ */
	//get splatmap dataset
	const auto& [tbo, tbo_count, reg, reg_count, dict, dict_count] = splat_texture;

	//prepare bindless texture
	this->SplatTextureHandle.reserve(tbo_count);
	for (unsigned int i = 0u; i < tbo_count; i++) {
		const GLuint64 handle = glGetTextureHandleARB(tbo[i]);
		this->SplatTextureHandle.emplace_back(handle);
		//active this handle
		glMakeTextureHandleResidentARB(handle);
	}

	//send bindless handle to the shader
	this->TerrainComponent.uniform(glProgramUniformHandleui64vARB, "RegionTexture", static_cast<GLsizei>(tbo_count), this->SplatTextureHandle.data())
		//prepare registry
		.uniform(glProgramUniform2uiv, "RegionRegistry", static_cast<GLsizei>(reg_count), reinterpret_cast<const unsigned int*>(reg))
		.uniform(glProgramUniform1uiv, "RegistryDictionary", static_cast<GLsizei>(dict_count), dict);
}

STPHeightfieldTerrain::~STPHeightfieldTerrain() {
	//deactivate bindless handles
	for (const auto handle : this->SplatTextureHandle) {
		glMakeTextureHandleNonResidentARB(handle);
	}
}

vec2 STPHeightfieldTerrain::calcBaseChunkPosition(const vec2& horizontal_offset) {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->TerrainGenerator.ChunkSetting;
	//calculate offset
	const ivec2 chunk_offset(-glm::floor(vec2(chunk_settings.RenderedChunk) / 2.0f));
	return STPChunk::offsetChunk(horizontal_offset, chunk_settings.ChunkSize, chunk_offset);
}

void STPHeightfieldTerrain::setMesh(const STPEnvironment::STPMeshSetting& mesh_setting) {
	//update tessellation LoD control
	this->TerrainComponent.uniform(glProgramUniform1f, "TessSetting.MaxLod", mesh_setting.TessSetting.MaxTessLevel)
		.uniform(glProgramUniform1f, "TessSetting.MinLod", mesh_setting.TessSetting.MinTessLevel)
		.uniform(glProgramUniform1f, "TessSetting.FurthestDistance", mesh_setting.TessSetting.FurthestTessDistance)
		.uniform(glProgramUniform1f, "TessSetting.NearestDistance", mesh_setting.TessSetting.NearestTessDistance)
		.uniform(glProgramUniform1f, "TessSetting.ShiftFactor", mesh_setting.LoDShiftFactor)
		//update other mesh-related parameters
		.uniform(glProgramUniform1f, "Altitude", mesh_setting.Altitude)
		.uniform(glProgramUniform1f, "NormalStrength", mesh_setting.Strength);
}

void STPHeightfieldTerrain::prepare(const vec3& viewPos) {
	//prepare heightfield
	this->TerrainGenerator.load(viewPos);

	//update the current view position
	this->ViewPosition = viewPos;
}

void STPHeightfieldTerrain::operator()() const {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	mat4 Model = glm::identity<mat4>();
	//move the terrain centre to the camera
	const vec2 chunkCentre = STPChunk::getChunkPosition(
		this->ViewPosition - chunk_setting.ChunkOffset,
		chunk_setting.ChunkSize,
		chunk_setting.ChunkScaling
	);
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
	this->TerrainComponent.uniform(glProgramUniformMatrix4fv, "MeshModel", 1, GL_FALSE, value_ptr(Model));

	//waiting for the heightfield generator to finish
	this->TerrainGenerator.wait();

	//prepare for rendering
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::BIOME]);
	glBindTextureUnit(1, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);
	glBindTextureUnit(2, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::SPLAT]);
	this->TileArray.bind();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
	this->TerrainRenderer.bind();
	
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPPipelineManager::unbind();
}