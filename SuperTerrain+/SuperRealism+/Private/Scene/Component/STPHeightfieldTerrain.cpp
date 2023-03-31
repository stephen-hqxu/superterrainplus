#include <SuperRealism+/Scene/Component/STPHeightfieldTerrain.h>
#include <SuperRealism+/STPRealismInfo.h>
//Noise Generator
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>
#include <SuperRealism+/Utility/STPRandomTextureGenerator.cuh>
//Chunk
#include <SuperTerrain+/World/Chunk/STPChunk.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>
//Indirect
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//GLAD
#include <glad/glad.h>

#include <array>
#include <limits>

//GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

using std::array;
using std::make_unique;
using std::unique_ptr;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::uvec3;
using glm::dvec3;
using glm::mat4;
using glm::dmat4;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto HeightfieldTerrainShaderFilename = SuperTerrainPlus::STPStringUtility::generateFilename(
	STPRealismInfo::ShaderPath, "/STPHeightfieldTerrain", ".vert", ".tesc", ".tese", ".frag");
//geometry shader is only used in shadow pass
constexpr static auto HeightfieldTerrainShadowShaderFilename = SuperTerrainPlus::STPStringUtility::generateFilename(
	STPRealismInfo::ShaderPath, "/STPHeightfieldTerrain", ".geom");

constexpr static STPIndirectCommand::STPDrawElement TerrainDrawCommand = {
	//Need to set the number of terrain tile during initialisation
	0u,
	1u,
	0u,
	0,
	0u
};

STPHeightfieldTerrain::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, const STPTerrainShaderOption& option) :
	TerrainGenerator(generator_pipeline), NoiseSample(GL_TEXTURE_3D) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	const STPDiversity::STPTextureFactory& splatmap_generator = this->TerrainGenerator.splatmapGenerator();
	const STPDiversity::STPTextureInformation::STPSplatTextureDatabase splat_texture = splatmap_generator.getSplatTexture();

	/* ---------------------------------------- setup terrain tile buffer ---------------------------------------- */
	const dvec2 chunkHorizontalOffset = dvec2(chunk_setting.ChunkOffset.x, chunk_setting.ChunkOffset.z);
	const uvec2 tileDimension = chunk_setting.ChunkSize * chunk_setting.RenderDistance;

	//generate terrain mesh
	this->TerrainMesh.emplace(tileDimension, this->calcBaseChunkPosition(chunkHorizontalOffset));
	//setup indirect buffer
	STPIndirectCommand::STPDrawElement cmd = TerrainDrawCommand;
	cmd.Count = this->TerrainMesh->planeIndexCount();
	this->TerrainRenderCommand.bufferStorageSubData(&cmd, sizeof(cmd), GL_NONE);

	/* ------------------------------------------ setup terrain shader ---------------------------------------------- */
	const size_t terrain_shader_count = HeightfieldTerrainShaderFilename.size();
	constexpr static GLenum terrain_shader_type[terrain_shader_count] = {
		GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
	STPShaderManager::STPShader terrain_shader[terrain_shader_count];

	for (unsigned int i = 0u; i < terrain_shader_count; i++) {
		const char* const source_file = HeightfieldTerrainShaderFilename[i].data();
		STPShaderManager::STPShaderSource shader_source(source_file, STPFile::read(source_file));

		if (i == 3u) {
			//fragment shader
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
			//prepare identifiers for texture splatting
			using namespace SuperTerrainPlus::STPDiversity;
			//general info
			Macro("GROUP_COUNT", splat_texture.TextureHandleCount)
				//The registry contains region that either has and has no texture data.
				("REGISTRY_COUNT", splat_texture.LocationRegistryDictionaryCount)
				("SPLAT_REGION_COUNT", splat_texture.SplatRegionCount)

			//texture type
			("ALBEDO", splatmap_generator.convertType(STPTextureType::Albedo))
			("NORMAL", splatmap_generator.convertType(STPTextureType::Normal))
			("ROUGHNESS", splatmap_generator.convertType(STPTextureType::Roughness))
			("AO", splatmap_generator.convertType(STPTextureType::AmbientOcclusion))

			("TYPE_STRIDE", splatmap_generator.usedType())
			("UNREGISTERED_TYPE", STPTextureFactory::UnregisteredType)

			("NORMALMAP_BLENDING", static_cast<std::underlying_type_t<STPNormalBlendingAlgorithm>>(option.NormalBlender));

			//process fragment shader
			shader_source.define(Macro);
		}
		//compile
		terrain_shader[i] = STPShaderManager::make(terrain_shader_type[i], shader_source);
	}

	//link programs
	STPProgramManager::STPProgramParameter terrain_program_option = { };
	terrain_program_option.Separable = true;
	//vertex shader
	this->TerrainVertex = STPProgramManager({ terrain_shader }, &terrain_program_option);
	//2 tessellation shaders
	this->TerrainModeller = STPProgramManager({ terrain_shader + 1, terrain_shader + 2 }, &terrain_program_option);
	//fragment shader
	this->TerrainShader = STPProgramManager({ terrain_shader + 3 }, &terrain_program_option);

	//build pipeline
	this->TerrainRenderer = STPPipelineManager({
		{ GL_VERTEX_SHADER_BIT, &this->TerrainVertex },
		{ GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, &this->TerrainModeller },
		{ GL_FRAGMENT_SHADER_BIT, &this->TerrainShader }
	});

	/* ------------------------------- setup initial immutable uniforms ---------------------------------- */
	//setup mesh model uniform location
	this->MeshModelLocation = this->TerrainVertex.uniformLocation("MeshModel");
	this->MeshQualityLocation = this->TerrainModeller.uniformLocation("TerrainRenderPass");

	//setup program for meshing the terrain
	this->TerrainModeller
		//heightfield for displacement mapping
		.uniform(glProgramUniform1i, "Heightfield", 0)
		//by default we use the high quality mesh for rendering
		.uniform(glProgramUniform1ui, this->MeshQualityLocation, 0u);

	//setup program that shades terrain with colour
	this->TerrainShader
		//some samplers
		.uniform(glProgramUniform1i, "Heightmap", 0)
		.uniform(glProgramUniform1i, "Splatmap", 1)
		//extra terrain info for rendering
		.uniform(glProgramUniform2uiv, "VisibleChunk", 1, value_ptr(chunk_setting.RenderDistance))
		.uniform(glProgramUniform2fv, "ChunkHorizontalOffset", 1, value_ptr(static_cast<vec2>(chunkHorizontalOffset)));

	/* --------------------------------- setup texture splatting ------------------------------------ */
	//get splatmap dataset
	const auto& [tbo_handle, tbo_handle_count, registry, registry_count, dict, dict_count, view_reg, region_count] = splat_texture;

	using STPDiversity::STPTextureInformation::STPTextureDataLocation;
	//prepare region registry
	//store all region registry data into a buffer and grab the device address
	this->SplatRegion.bufferStorageSubData(registry, sizeof(STPTextureDataLocation) * registry_count, GL_NONE);
	this->SplatRegionAddress = this->SplatRegion.getAddress();
	this->SplatRegion.makeResident(GL_READ_ONLY);

	//next build the registry lookup dictionary
	unique_ptr<GLuint64EXT[]> region_data_address = make_unique<GLuint64EXT[]>(dict_count);
	for (unsigned int i = 0u; i < dict_count; i++) {
		const unsigned int reg_loc = dict[i];
		//some region might have no associated texture data, therefore we assign a null pointer
		if (reg_loc >= registry_count) {
			region_data_address[i] = 0ull;
			continue;
		}

		//if this region has texture data, we lookup the address of it.
		region_data_address[i] = this->SplatRegionAddress + sizeof(STPTextureDataLocation) * reg_loc;
	}

	//prepare texture region scaling data
	unique_ptr<uvec3[]> scale_factors = make_unique<uvec3[]>(region_count);
	std::transform(view_reg, view_reg + region_count, scale_factors.get(), [](const auto& tex_view) {
		return uvec3(tex_view.PrimaryScale, tex_view.SecondaryScale, tex_view.TertiaryScale);
	});
	//send bindless handle to the shader
	this->TerrainShader.uniform(glProgramUniformHandleui64vARB, "RegionTexture", static_cast<GLsizei>(tbo_handle_count), tbo_handle)
		//prepare registry
		.uniform(glProgramUniformui64vNV, "RegionRegistry", static_cast<GLsizei>(dict_count), region_data_address.get())
		.uniform(glProgramUniform3uiv, "RegionScaleRegistry", static_cast<GLsizei>(region_count), value_ptr(scale_factors[0]));

	/* --------------------------------- shader noise texture preparation ------------------------------ */
	this->NoiseSample.textureStorage3D(1, GL_R8, option.NoiseDimension);
	this->NoiseSample.wrap(GL_REPEAT);
	this->NoiseSample.filter(GL_NEAREST, GL_LINEAR);
	{
		//generate noise texture
		using namespace SuperTerrainPlus::STPSmartDeviceObject;
		using std::numeric_limits;

		STPStream managed_stream = makeStream(cudaStreamNonBlocking);
		const cudaStream_t stream = managed_stream.get();

		STPRandomTextureGenerator::generate(this->NoiseSample, option.NoiseSeed, numeric_limits<unsigned char>::min(),
			numeric_limits<unsigned char>::max(), stream);

		STP_CHECK_CUDA(cudaStreamSynchronize(stream));
	}
	//create bindless handle for noise sampler
	this->NoiseSampleHandle = STPBindlessTexture::make(this->NoiseSample);
	this->TerrainShader.uniform(glProgramUniformHandleui64ARB, "Noisemap", this->NoiseSampleHandle.get());

	/* -------------------------------------------- initialise ----------------------------------------- */
	this->TerrainGenerator.load(option.InitialViewPosition);
	//force update the model matrix
	this->updateTerrainModel();
}

dvec2 STPHeightfieldTerrain::calcBaseChunkPosition(const dvec2 horizontal_offset) {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->TerrainGenerator.ChunkSetting;
	//calculate offset
	const ivec2 base_chunk_coord = STPChunk::calcLocalChunkOrigin(ivec2(0), chunk_settings.ChunkSize, chunk_settings.RenderDistance);

	return static_cast<dvec2>(base_chunk_coord) + horizontal_offset;
}

inline void STPHeightfieldTerrain::updateTerrainModel() {
	//update model matrix
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	dmat4 Model = glm::identity<dmat4>();
	//move the terrain centre to the camera
	const ivec2& chunkCentre = this->TerrainGenerator.centre();
	Model = glm::scale(Model, dvec3(
		chunk_setting.ChunkScale.x,
		1.0,
		chunk_setting.ChunkScale.y
	));
	Model = glm::translate(Model, dvec3(
		chunkCentre.x + chunk_setting.ChunkOffset.x,
		chunk_setting.ChunkOffset.y,
		chunkCentre.y + chunk_setting.ChunkOffset.z
	));

	//update the current model matrix
	//use double precision for intermediate calculation to avoid rounding errors, and cast to single precision.
	this->TerrainVertex.uniform(glProgramUniformMatrix4fv, this->MeshModelLocation, 1, 
		static_cast<GLboolean>(GL_FALSE), value_ptr(static_cast<mat4>(Model)));
}

void STPHeightfieldTerrain::setMesh(const STPEnvironment::STPMeshSetting& mesh_setting) {
	mesh_setting.validate();
	const auto& tess_setting = mesh_setting.TessSetting;
	const auto& smooth_setting = mesh_setting.RegionSmoothSetting;
	const auto& scale_setting = mesh_setting.RegionScaleSetting;

	//update tessellation LoD control
	this->TerrainModeller.uniform(glProgramUniform1f, "Tess[0].MaxLod", tess_setting.MaxTessLevel)
		.uniform(glProgramUniform1f, "Tess[0].MinLod", tess_setting.MinTessLevel)
		.uniform(glProgramUniform1f, "Tess[0].MaxDis", tess_setting.FurthestTessDistance)
		.uniform(glProgramUniform1f, "Tess[0].MinDis", tess_setting.NearestTessDistance)
		//set default terrain rendering pass to regular rendering
		.uniform(glProgramUniform1ui, "TerrainRenderPass", 0u)
		//update other mesh-related parameters
		.uniform(glProgramUniform1f, "Altitude", mesh_setting.Altitude);

	//update settings for rendering
	this->TerrainShader.uniform(glProgramUniform1f, "NormalStrength", mesh_setting.Strength)
		//texture splatting smoothing
		.uniform(glProgramUniform1ui, "SmoothSetting.Kr", smooth_setting.KernelRadius)
		.uniform(glProgramUniform1f, "SmoothSetting.Ks", smooth_setting.KernelScale)
		.uniform(glProgramUniform1ui, "SmoothSetting.Ns", smooth_setting.NoiseScale)
		//texture region scaling
		.uniform(glProgramUniform1f, "ScaleSetting.Prim", scale_setting.PrimaryFar)
		.uniform(glProgramUniform1f, "ScaleSetting.Seco", scale_setting.SecondaryFar)
		.uniform(glProgramUniform1f, "ScaleSetting.Tert", scale_setting.TertiaryFar);
}

void STPHeightfieldTerrain::setDepthMeshQuality(const STPEnvironment::STPTessellationSetting& tess) {
	this->TerrainModeller.uniform(glProgramUniform1f, "Tess[1].MaxLod", tess.MaxTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MinLod", tess.MinTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MaxDis", tess.FurthestTessDistance)
		.uniform(glProgramUniform1f, "Tess[1].MinDis", tess.NearestTessDistance);
}

void STPHeightfieldTerrain::setViewPosition(const dvec3& viewPos) {
	//prepare heightfield
	if (this->TerrainGenerator.load(viewPos) != STPWorldPipeline::STPWorldLoadStatus::Swapped) {
		//centre chunk has yet changed, nothing to do.
		return;
	}

	this->updateTerrainModel();
}

void STPHeightfieldTerrain::render() const {
	//prepare for rendering
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPTerrainMapType::Heightmap]);
	glBindTextureUnit(1, this->TerrainGenerator[STPWorldPipeline::STPTerrainMapType::Splatmap]);

	this->TerrainMesh->bindPlaneVertexArray();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	this->TerrainRenderer.bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);

	//clear up
	STPPipelineManager::unbind();
}

bool STPHeightfieldTerrain::addDepthConfiguration(const size_t light_space_count, const STPShaderManager::STPShader* const depth_shader) {
	//create a new render group
	auto [depth_group, inserted] = this->TerrainDepthRenderer.try_emplace(light_space_count);
	if (!inserted) {
		//group exists, don't add
		return false;
	}
	auto& [depth_renderer, depth_writer_arr] = depth_group->second;
	auto& [depth_writer] = depth_writer_arr;

	//geometry shader for depth writing
	//make a copy of the original source because we need to modify it
	const char* const shadow_source_file = HeightfieldTerrainShadowShaderFilename.data();
	STPShaderManager::STPShaderSource shadow_shader_source(shadow_source_file, STPFile::read(shadow_source_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	//disable eval shader output because shadow pass does need those data
	Macro("HEIGHTFIELD_SHADOW_PASS_INVOCATION", light_space_count);
	shadow_shader_source.define(Macro);

	//now the base renderer is finished, setup depth renderer
	const STPShaderManager::STPShader terrain_shadow_shader = STPShaderManager::make(GL_GEOMETRY_SHADER, shadow_shader_source);

	STPProgramManager::STPProgramParameter depth_option = { };
	depth_option.Separable = true;
	//link program for depth writing
	if (depth_shader) {
		depth_writer = STPProgramManager({ &terrain_shadow_shader, depth_shader }, &depth_option);
	} else {
		depth_writer = STPProgramManager({ &terrain_shadow_shader }, &depth_option);
	}

	//build shadow pipeline
	depth_renderer = STPPipelineManager({
		{ GL_VERTEX_SHADER_BIT,	&this->TerrainVertex },
		{ GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, &this->TerrainModeller },
		{ GL_GEOMETRY_SHADER_BIT | (depth_shader ? GL_FRAGMENT_SHADER_BIT : 0), &depth_writer }
	});

	return true;
}

void STPHeightfieldTerrain::renderDepth(const size_t light_space_count) const {
	//in this case we only need heightfield for tessellation
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPTerrainMapType::Heightmap]);

	this->TerrainMesh->bindPlaneVertexArray();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
	//enable low quality mesh
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 1u);

	//find the correct render group
	this->TerrainDepthRenderer.at(light_space_count).first.bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);

	//clear up
	STPPipelineManager::unbind();
	//change back to normal rendering
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 0u);
}