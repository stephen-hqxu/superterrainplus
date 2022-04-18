#include <SuperRealism+/Scene/Component/STPWater.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

#include <SuperRealism+/Object/STPShaderManager.h>

//GLAD
#include <glad/glad.h>

#include <memory>
#include <algorithm>

//GLM
#include <glm/common.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/exponential.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::unique_ptr;
using std::make_unique;

using glm::uvec3;
using glm::ivec3;
using glm::vec4;
using glm::value_ptr;

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

//Water shader shares a majority part with the terrain shader.
constexpr static auto WaterShaderFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPHeightfieldTerrain", ".tesc", ".tese");
constexpr static auto WaterFragmentShaderFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPWater", ".frag");
constexpr static size_t WaterShaderCount = 3ull;

STPWater::STPWater(const STPHeightfieldTerrain<false>& terrain, const STPBiomeWaterLevel& water_level) : TerrainObject(terrain), WaterLevelTable(GL_TEXTURE_1D), 
	WavePhase(0.0), WavePeriod(0.0) {
	/* ----------------------------- setup water shader -------------------------------- */
	STPShaderManager water_shader[WaterShaderCount] = {
		GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER
	};
	for (unsigned int i = 0u; i < WaterShaderCount; i++) {
		const char* const water_source_file = i < 2u ? WaterShaderFilename[i].data() : WaterFragmentShaderFilename.data();
		STPShaderManager::STPShaderSource water_source(water_source_file, *STPFile(water_source_file));

		if (i < 2u) {
			//tessellation shaders
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

			//enable water shader routine
			Macro("STP_WATER", 1);

			water_source.define(Macro);
		}
		water_shader[i](water_source);

		//attach
		this->WaterAnimator.attach(water_shader[i]);
	}
	//link
	this->WaterAnimator.separable(true);
	this->WaterAnimator.finalise();

	this->WaterRenderer
		//water shares the same vertex program (and hence the mesh model) with the terrain
		.stage(GL_VERTEX_SHADER_BIT, this->TerrainObject.TerrainVertex)
		.stage(GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT | GL_FRAGMENT_SHADER_BIT, this->WaterAnimator)
		.finalise();

	/* --------------------------------- build water level dictionary -------------------------- */
	//default water level for biome with no given water level
	constexpr static float DefaultWaterLevel = -1.0f;
	//convert hash table to a sorted array
	//find the max biome ID in the table
	const unsigned int biomeCount = std::max_element(water_level.cbegin(), water_level.cend(), 
		[](auto v1, auto v2) { return v1.first < v2.first; })->first + 1u;

	//now we need to expand this table so every biome can be looked up
	unique_ptr<float[]> waterLevelDict = make_unique<float[]>(biomeCount);
	for (unsigned int i = 0u; i < biomeCount; i++) {
		//find water level at the biome
		auto biome_it = water_level.find(static_cast<STPDiversity::Sample>(i));
		//biome not found, fill a zero; or set the value
		waterLevelDict[i] = (biome_it == water_level.cend()) ? DefaultWaterLevel : biome_it->second;
	}
	//now we can use biome ID as an index to the water level dictionary, and look up the water level

	//setup water level lookup table
	const uvec3 table_size = uvec3(biomeCount, 1u, 1u);
	this->WaterLevelTable.textureStorage<STPTexture::STPDimension::ONE>(1, GL_R16, table_size);
	this->WaterLevelTable.textureSubImage<STPTexture::STPDimension::ONE>(0, ivec3(0), table_size, GL_RED, GL_FLOAT, waterLevelDict.get());

	this->WaterLevelTable.filter(GL_NEAREST, GL_NEAREST);
	//we use clamp to border so if a biome ID that is greater than the size of the table will give a valid value
	this->WaterLevelTable.wrap(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
	this->WaterLevelTable.borderColor(vec4(DefaultWaterLevel));
	
	//create bindless handle
	this->WaterLevelTableHandle.emplace(this->WaterLevelTable);

	/* ----------------------------------- water uniform setup --------------------------------- */
	this->WaterAnimator.uniform(glProgramUniform1i, "Biomemap", 0)
		.uniform(glProgramUniformHandleui64ARB, "WaterLevel", **this->WaterLevelTableHandle);

	this->WaveTimeLocation = this->WaterAnimator.uniformLocation("WaveTime");
}

void STPWater::setWater(const STPEnvironment::STPWaterSetting& water_setting) {
	if (!water_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Water setting fails to validate");
	}
	const auto& wave_setting = water_setting.WaterWave;
	const auto& tess_setting = water_setting.WaterMeshTess;

	//find a common wave period for each wave function
	const auto [it_geo, it_norm] = water_setting.WaterWaveIteration;
	const unsigned int iteration = glm::min(it_geo, it_norm);
	//calculate the wave frequency after applying all octaves
	const double initFreq = static_cast<double>(wave_setting.InitialFrequency),
		finalFreq = initFreq * glm::pow(wave_setting.Lacunarity, iteration),
		//this is the common frequency for both waves
		dominantFreq = glm::min(initFreq, finalFreq);
	
	//reset wave parameters
	this->WavePhase = 0.0;
	this->WavePeriod = 2.0 * glm::pi<double>() / dominantFreq;

	/* ---------------------------- send data ----------------------------------- */
	//water plane tessellation settings
	this->WaterAnimator.uniform(glProgramUniform1f, "WaterTess.MaxLod", tess_setting.MaxTessLevel)
		.uniform(glProgramUniform1f, "WaterTess.MinLod", tess_setting.MinTessLevel)
		.uniform(glProgramUniform1f, "WaterTess.MaxDis", tess_setting.FurthestTessDistance)
		.uniform(glProgramUniform1f, "WaterTess.MinDis", tess_setting.NearestTessDistance)
		//water plane culling test
		.uniform(glProgramUniform1f, "MinLevel", water_setting.MinimumWaterLevel)
		.uniform(glProgramUniform1ui, "SampleCount", water_setting.CullTestSample)
		.uniform(glProgramUniform1f, "SampleRadiusMul", water_setting.CullTestRadius)
		.uniform(glProgramUniform1f, "Altitude", water_setting.Altitude)
		//water wave animate function
		.uniform(glProgramUniform1f, "WaveHeight", water_setting.WaveHeight)
		.uniform(glProgramUniform1ui, "WaveGeometryIteration", it_geo)
		.uniform(glProgramUniform1ui, "WaveNormalIteration", it_norm)
		//water shading
		.uniform(glProgramUniform1f, "Epsilon", water_setting.NormalEpsilon)
		.uniform(glProgramUniform3fv, "Tint", 1, value_ptr(water_setting.Tint))
		//wave function
		.uniform(glProgramUniform1f, "WaterWave.iRot", wave_setting.InitialRotation)
		.uniform(glProgramUniform1f, "WaterWave.iFreq", wave_setting.InitialFrequency)
		.uniform(glProgramUniform1f, "WaterWave.iAmp", wave_setting.InitialAmplitude)
		.uniform(glProgramUniform1f, "WaterWave.iSpd", wave_setting.InitialSpeed)
		.uniform(glProgramUniform1f, "WaterWave.octRot", wave_setting.OctaveRotation)
		.uniform(glProgramUniform1f, "WaterWave.Lacu", wave_setting.Lacunarity)
		.uniform(glProgramUniform1f, "WaterWave.Pers", wave_setting.Persistence)
		.uniform(glProgramUniform1f, "WaterWave.octSpd", wave_setting.OctaveSpeed)
		.uniform(glProgramUniform1f, "WaterWave.Drag", wave_setting.WaveDrag);

	//flush the new wave time
	this->advanceWaveTime(0.0);
}

double STPWater::getWavePeriod() const {
	return this->WavePeriod;
}

void STPWater::advanceWaveTime(double time) {
	this->WavePhase += time;
	//wrap the wave around
	if (this->WavePhase >= this->WavePeriod) {
		this->WavePhase -= this->WavePeriod;
	}

	this->WaterAnimator.uniform(glProgramUniform1f, this->WaveTimeLocation, static_cast<float>(this->WavePhase));
}

void STPWater::render() const {
	STPWorldPipeline& world_gen = this->TerrainObject.TerrainGenerator;
	//water as a transparent object is always rendered before opaque object like terrain.
	//so may be sync is not necessary?
	world_gen.wait();

	//prepare for texture
	glBindTextureUnit(0, world_gen[STPWorldPipeline::STPRenderingBufferType::BIOME]);
	//prepare for plane data, which shares with the terrain
	this->TerrainObject.TerrainMesh->bindPlaneVertexArray();
	this->TerrainObject.TerrainRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	this->WaterRenderer.bind();
	//render
	glDrawElementsIndirect(GL_PATCHES, GL_UNSIGNED_INT, nullptr);

	STPPipelineManager::unbind();
}