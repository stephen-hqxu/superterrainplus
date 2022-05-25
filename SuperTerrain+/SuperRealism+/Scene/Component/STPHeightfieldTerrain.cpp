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
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

using std::array;
using std::numeric_limits;
using std::unique_ptr;
using std::make_unique;

using glm::ivec2;
using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::uvec3;
using glm::dvec3;
using glm::mat4;
using glm::dmat4;
using glm::value_ptr;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto HeightfieldTerrainShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPHeightfieldTerrain", 
	".vert", ".tesc", ".tese", ".frag");
//geometry shader is only used in shadow pass
constexpr static auto HeightfieldTerrainShadowShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPHeightfieldTerrain", ".geom");

constexpr static STPIndirectCommand::STPDrawElement TerrainDrawCommand = {
	//Need to set the number of terrain tile during initialisation
	0u,
	1u,
	0u,
	0u,
	0u
};

STPHeightfieldTerrain<false>::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, const STPTerrainShaderOption& option) :
	TerrainGenerator(generator_pipeline), NoiseSample(GL_TEXTURE_3D), RandomTextureDimension(option.NoiseDimension) {
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	const STPDiversity::STPTextureFactory& splatmap_generator = this->TerrainGenerator.splatmapGenerator();
	const STPDiversity::STPTextureInformation::STPSplatTextureDatabase splat_texture = splatmap_generator.getSplatTexture();

	/* ---------------------------------------- setup terrain tile buffer ---------------------------------------- */
	const dvec2 chunkHorizontalOffset = dvec2(chunk_setting.ChunkOffset.x, chunk_setting.ChunkOffset.z);
	const uvec2 tileDimension = chunk_setting.ChunkSize * chunk_setting.RenderedChunk;

	//generate terrain mesh
	this->TerrainMesh.emplace(tileDimension, this->calcBaseChunkPosition(chunkHorizontalOffset));
	//setup indirect buffer
	STPIndirectCommand::STPDrawElement cmd = TerrainDrawCommand;
	cmd.Count = this->TerrainMesh->planeIndexCount();
	this->TerrainRenderCommand.bufferStorageSubData(&cmd, sizeof(cmd), GL_NONE);

	/* ------------------------------------------ setup terrain shader ---------------------------------------------- */
	STPShaderManager terrain_shader[HeightfieldTerrainShaderFilename.size()] = {
		GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER
	};
	for (unsigned int i = 0u; i < HeightfieldTerrainShaderFilename.size(); i++) {
		const char* const source_file = HeightfieldTerrainShaderFilename[i].data();
		STPShaderManager::STPShaderSource shader_source(source_file, *STPFile(source_file));

		if (i == 3u) {
			//fragment shader
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
			//prepare identifiers for texture splatting
			using namespace SuperTerrainPlus::STPDiversity;
			//general info
			Macro("GROUP_COUNT", splat_texture.TextureBufferCount)
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
		terrain_shader[i](shader_source);

		//attach the complete terrain program
		switch (i) {
		case 0u:
			//vertex shader
			this->TerrainVertex.attach(terrain_shader[i]);
			break;
		case 3u:
			//fragment shader
			this->TerrainShader.attach(terrain_shader[i]);
			break;
		default:
			//anything else
			this->TerrainModeller.attach(terrain_shader[i]);
			break;
		}
	}
	this->TerrainVertex.separable(true);
	this->TerrainModeller.separable(true);
	this->TerrainShader.separable(true);

	//link
	this->TerrainVertex.finalise();
	this->TerrainModeller.finalise();
	this->TerrainShader.finalise();

	//build pipeline
	this->TerrainRenderer
		.stage(GL_VERTEX_SHADER_BIT, this->TerrainVertex)
		.stage(GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, this->TerrainModeller)
		.stage(GL_FRAGMENT_SHADER_BIT, this->TerrainShader)
		.finalise();

	/* ------------------------------- setup initial immutable uniforms ---------------------------------- */
	//setup mesh model uniform location
	this->MeshModelLocation = this->TerrainVertex.uniformLocation("MeshModel");

	//setup program for meshing the terrain
	this->TerrainModeller
		//heightfield for displacement mapping
		.uniform(glProgramUniform1i, "Heightfield", 0);

	//setup program that shades terrain with colour
	this->TerrainShader
		//some samplers
		.uniform(glProgramUniform1i, "Heightmap", 0)
		.uniform(glProgramUniform1i, "Splatmap", 1)
		//extra terrain info for rendering
		.uniform(glProgramUniform2uiv, "VisibleChunk", 1, value_ptr(chunk_setting.RenderedChunk))
		.uniform(glProgramUniform2fv, "ChunkHorizontalOffset", 1, value_ptr(static_cast<vec2>(chunkHorizontalOffset)));

	/* --------------------------------- setup texture splatting ------------------------------------ */
	//get splatmap dataset
	const auto& [tbo, tbo_count, registry, registry_count, dict, dict_count, view_reg, region_count] = splat_texture;

	using STPDiversity::STPTextureInformation::STPTextureDataLocation;
	//prepare region registry
	//store all region registry data into a buffer and grab the device address
	this->SplatRegion.bufferStorageSubData(registry, sizeof(STPTextureDataLocation) * registry_count, GL_NONE);
	this->SplatRegionAddress = STPBindlessBuffer(this->SplatRegion, GL_READ_ONLY);
	const GLuint64EXT region_address_beg = *this->SplatRegionAddress;

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
		region_data_address[i] = region_address_beg + sizeof(STPTextureDataLocation) * reg_loc;
	}

	//prepare bindless texture
	this->SplatTextureHandle.reserve(tbo_count);
	unique_ptr<GLuint64[]> rawHandle = make_unique<GLuint64[]>(tbo_count);
	for (unsigned int i = 0u; i < tbo_count; i++) {
		//extract raw handle so we can send them to the shader via uniform
		rawHandle[i] = *this->SplatTextureHandle.emplace_back(tbo[i]);
	}

	//prepare texture region scaling data
	unique_ptr<uvec3[]> scale_factors = make_unique<uvec3[]>(region_count);
	std::transform(view_reg, view_reg + region_count, scale_factors.get(), [](const auto& tex_view) {
		return uvec3(tex_view.PrimaryScale, tex_view.SecondaryScale, tex_view.TertiaryScale);
	});
	//send bindless handle to the shader
	this->TerrainShader.uniform(glProgramUniformHandleui64vARB, "RegionTexture", static_cast<GLsizei>(tbo_count), rawHandle.get())
		//prepare registry
		.uniform(glProgramUniformui64vNV, "RegionRegistry", static_cast<GLsizei>(dict_count), region_data_address.get())
		.uniform(glProgramUniform3uiv, "RegionScaleRegistry", static_cast<GLsizei>(region_count), value_ptr(scale_factors[0]));

	/* --------------------------------- shader noise texture preparation ------------------------------ */
	this->NoiseSample.textureStorage<STPTexture::STPDimension::THREE>(1, GL_R8, this->RandomTextureDimension);
	this->NoiseSample.wrap(GL_REPEAT);
	this->NoiseSample.filter(GL_NEAREST, GL_LINEAR);
}

dvec2 STPHeightfieldTerrain<false>::calcBaseChunkPosition(dvec2 horizontal_offset) {
	const STPEnvironment::STPChunkSetting& chunk_settings = this->TerrainGenerator.ChunkSetting;
	//calculate offset
	const ivec2 chunk_offset = -static_cast<ivec2>(chunk_settings.RenderedChunk / 2u),
		base_chunk_coord = STPChunk::offsetChunk(ivec2(0), chunk_settings.ChunkSize, chunk_offset);

	return static_cast<dvec2>(base_chunk_coord) + horizontal_offset;
}

void STPHeightfieldTerrain<false>::setMesh(const STPEnvironment::STPMeshSetting& mesh_setting) {
	if (!mesh_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Mesh setting is not validated");
	}
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

void STPHeightfieldTerrain<false>::seedRandomBuffer(unsigned long long seed) {
	cudaGraphicsResource_t res;
	cudaArray_t random_buffer;

	//CUDA will throw error when mapping on a texture with bindless handle active, so we need to deactivate it first.
	this->NoiseSampleHandle.~STPBindlessTexture();
	//register CUDA graphics
	STPcudaCheckErr(cudaGraphicsGLRegisterImage(&res, *this->NoiseSample, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsWriteDiscard));
	//map
	STPcudaCheckErr(cudaGraphicsMapResources(1, &res));
	STPcudaCheckErr(cudaGraphicsSubResourceGetMappedArray(&random_buffer, res, 0u, 0u));

	//compute
	STPRandomTextureGenerator::generate<unsigned char>(random_buffer, this->RandomTextureDimension, seed, 
		numeric_limits<unsigned char>::min(), numeric_limits<unsigned char>::max());

	//clear up
	STPcudaCheckErr(cudaGraphicsUnmapResources(1, &res));
	STPcudaCheckErr(cudaGraphicsUnregisterResource(res));

	//create bindless handle for noise sampler
	this->NoiseSampleHandle = STPBindlessTexture(this->NoiseSample);
	this->TerrainShader.uniform(glProgramUniformHandleui64ARB, "Noisemap", *this->NoiseSampleHandle);
}

void STPHeightfieldTerrain<false>::setViewPosition(const dvec3& viewPos) {
	//prepare heightfield
	if (!this->TerrainGenerator.load(viewPos)) {
		//centre chunk has yet changed, nothing to do.
		return;
	}

	//update model matrix
	const STPEnvironment::STPChunkSetting& chunk_setting = this->TerrainGenerator.ChunkSetting;
	dmat4 Model = glm::identity<dmat4>();
	//move the terrain centre to the camera
	const ivec2& chunkCentre = this->TerrainGenerator.centre();
	Model = glm::scale(Model, dvec3(
		chunk_setting.ChunkScaling,
		1.0f,
		chunk_setting.ChunkScaling
	));
	Model = glm::translate(Model, dvec3(
		chunkCentre.x + chunk_setting.ChunkOffset.x,
		chunk_setting.ChunkOffset.y,
		chunkCentre.y + chunk_setting.ChunkOffset.z
	));

	//update the current model matrix
	//use double precision for intermediate calculation to avoid rounding errors, and cast to single precision.
	this->TerrainVertex.uniform(glProgramUniformMatrix4fv, this->MeshModelLocation, 1, static_cast<GLboolean>(GL_FALSE), value_ptr(static_cast<mat4>(Model)));
}

void STPHeightfieldTerrain<false>::render() const {
	//waiting for the heightfield generator to finish
	this->TerrainGenerator.wait();

	//prepare for rendering
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);
	glBindTextureUnit(1, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::SPLAT]);

	this->TerrainMesh->bindPlaneVertexArray();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	this->TerrainRenderer.bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);

	//clear up
	STPPipelineManager::unbind();
}

STPHeightfieldTerrain<true>::STPHeightfieldTerrain(STPWorldPipeline& generator_pipeline, const STPTerrainShaderOption& option) :
	STPHeightfieldTerrain<false>(generator_pipeline, option), MeshQualityLocation(this->TerrainModeller.uniformLocation("TerrainRenderPass")) {

}

void STPHeightfieldTerrain<true>::setDepthMeshQuality(const STPEnvironment::STPTessellationSetting& tess) {
	this->TerrainModeller.uniform(glProgramUniform1f, "Tess[1].MaxLod", tess.MaxTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MinLod", tess.MinTessLevel)
		.uniform(glProgramUniform1f, "Tess[1].MaxDis", tess.FurthestTessDistance)
		.uniform(glProgramUniform1f, "Tess[1].MinDis", tess.NearestTessDistance);
}

bool STPHeightfieldTerrain<true>::addDepthConfiguration(size_t light_space_count, const STPShaderManager* depth_shader) {
	//create a new render group
	if (this->TerrainDepthRenderer.exist(light_space_count)) {
		//group exists, don't add
		return false;
	}
	auto& [depth_renderer, depth_writer_arr] = this->TerrainDepthRenderer.addGroup(light_space_count);
	auto& [depth_writer] = depth_writer_arr;

	//now the base renderer is finished, setup depth renderer
	STPShaderManager terrain_shadow_shader(GL_GEOMETRY_SHADER);

	//geometry shader for depth writing
	//make a copy of the original source because we need to modify it
	const char* const shadow_source_file = HeightfieldTerrainShadowShaderFilename.data();
	STPShaderManager::STPShaderSource shadow_shader_source(shadow_source_file, *STPFile(shadow_source_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	//disable eval shader output because shadow pass does need those data
	Macro("HEIGHTFIELD_SHADOW_PASS_INVOCATION", light_space_count);

	shadow_shader_source.define(Macro);
	terrain_shadow_shader(shadow_shader_source);

	//attach program for depth writing
	if (depth_shader) {
		depth_writer.attach(*depth_shader);
	}
	depth_writer.attach(terrain_shadow_shader)
		.separable(true);

	//link
	depth_writer.finalise();

	//build shadow pipeline
	depth_renderer
		.stage(GL_VERTEX_SHADER_BIT, this->TerrainVertex)
		.stage(GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT, this->TerrainModeller)
		.stage(GL_GEOMETRY_SHADER_BIT | (depth_shader ? GL_FRAGMENT_SHADER_BIT : 0), depth_writer)
		.finalise();

	return true;
}

void STPHeightfieldTerrain<true>::renderDepth(size_t light_space_count) const {
	this->TerrainGenerator.wait();

	//in this case we only need heightfield for tessellation
	glBindTextureUnit(0, this->TerrainGenerator[STPWorldPipeline::STPRenderingBufferType::HEIGHTFIELD]);

	this->TerrainMesh->bindPlaneVertexArray();
	this->TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
	//enable low quality mesh
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 1u);

	//find the correct render group
	this->TerrainDepthRenderer.findPipeline(light_space_count).bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);

	//clear up
	STPPipelineManager::unbind();
	//change back to normal rendering
	this->TerrainModeller.uniform(glProgramUniform1ui, this->MeshQualityLocation, 0u);
}