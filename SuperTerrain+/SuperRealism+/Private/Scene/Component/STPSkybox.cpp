#include <SuperRealism+/Scene/Component/STPSkybox.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPValidationFailed.h>

#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

//GLAD
#include <glad/glad.h>

//Container
#include <array>

using std::array;
using std::shared_ptr;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SkyboxShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPSkybox", ".vert");

constexpr static array<signed char, 24u> BoxVertex = { 
	-1, -1, -1, //origin
	+1, -1, -1, //x=1
	+1, -1, +1, //x=z=1
	-1, -1, +1, //z=1
	-1, +1, -1, //y=1
	+1, +1, -1, //x=y=1
	+1, +1, +1, //x=y=z=1
	-1, +1, +1  //y=z=1
};
constexpr static array<unsigned char, 36u> BoxIndex = {
	0, 1, 2,
	0, 2, 3,

	0, 1, 5,
	0, 5, 4,

	1, 2, 6,
	1, 6, 5,

	2, 3, 7,
	2, 7, 6,

	3, 0, 4,
	3, 4, 7,

	4, 5, 6,
	4, 6, 7
};
constexpr static STPIndirectCommand::STPDrawElement BoxDrawCommand = {
	static_cast<unsigned int>(BoxIndex.size()),
	1u,
	0u,
	0,
	0u
};

STPSkybox::STPSkyboxVertexBuffer::STPSkyboxVertexBuffer() noexcept {
	//setup sky rendering buffer
	this->SkyboxBuffer.bufferStorageSubData(BoxVertex.data(), BoxVertex.size() * sizeof(signed char), GL_NONE);
	this->SkyboxIndex.bufferStorageSubData(BoxIndex.data(), BoxIndex.size() * sizeof(unsigned char), GL_NONE);
	//setup rendering command
	this->SkyboxDrawCommand.bufferStorageSubData(&BoxDrawCommand, sizeof(BoxDrawCommand), GL_NONE);
	
	//setup vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->SkyboxArray.attribute();
	attr.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->SkyboxBuffer, 0)
		.elementBuffer(this->SkyboxIndex)
		.binding();
	this->SkyboxArray.enable(0u);
}

void STPSkybox::STPSkyboxVertexBuffer::bind() const noexcept {
	this->SkyboxArray.bind();
	this->SkyboxDrawCommand.bind(GL_DRAW_INDIRECT_BUFFER);
}

inline static STPShaderManager::STPShader createSkyboxVertexShader() {
	const char* const skybox_source_file = SkyboxShaderFilename.data();
	STPShaderManager::STPShaderSource skybox_source(skybox_source_file, SuperTerrainPlus::STPFile::read(skybox_source_file));
	return STPShaderManager::make(GL_VERTEX_SHADER, skybox_source);
}

STPSkybox::STPSkyboxInitialiser::STPSkyboxInitialiser() : VertexShader(createSkyboxVertexShader()) {
	
}

void STPSkybox::initSkyboxRenderer(const STPShaderManager::STPShader& skybox_fs, const STPSkyboxInitialiser& skybox_init) {
	STP_ASSERTION_VALIDATION(STPShaderManager::shaderType(skybox_fs) == GL_FRAGMENT_SHADER,
		"The shader initialised for skybox rendering must a fragment shader");
	const auto& [skybox_vs, skybox_buf] = skybox_init;

	//construction of shared pointer throws exception if weak pointer is empty
	this->SkyboxVertex = shared_ptr(skybox_buf);

	this->SkyboxRenderer = STPProgramManager({ &skybox_vs, &skybox_fs });
}

void STPSkybox::drawSkybox() const noexcept {
	//bind
	this->SkyboxVertex->bind();
	const STPProgramManager::STPProgramStateManager skybox_state = this->SkyboxRenderer.useManaged();

	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);
}