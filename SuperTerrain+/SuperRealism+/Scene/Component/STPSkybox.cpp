#include <SuperRealism+/Scene/Component/STPSkybox.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPInvalidArgument.h>

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

constexpr static array<signed char, 24ull> BoxVertex = { 
	-1, -1, -1, //origin
	+1, -1, -1, //x=1
	+1, -1, +1, //x=z=1
	-1, -1, +1, //z=1
	-1, +1, -1, //y=1
	+1, +1, -1, //x=y=1
	+1, +1, +1, //x=y=z=1
	-1, +1, +1  //y=z=1
};
constexpr static array<unsigned char, 36ull> BoxIndex = {
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

STPSkybox::STPSkyboxVertexShader::STPSkyboxVertexShader() : SkyboxVertexShader(GL_VERTEX_SHADER) {
	const char* const skybox_source_file = SkyboxShaderFilename.data();
	STPShaderManager::STPShaderSource skybox_source(skybox_source_file, STPFile::read(skybox_source_file));

	//compile
	this->SkyboxVertexShader(skybox_source);
}

const STPShaderManager& STPSkybox::STPSkyboxVertexShader::operator*() const {
	return this->SkyboxVertexShader;
}

STPSkybox::STPSkyboxVertexBuffer::STPSkyboxVertexBuffer() {
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

void STPSkybox::STPSkyboxVertexBuffer::bind() const {
	this->SkyboxArray.bind();
	this->SkyboxDrawCommand.bind(GL_DRAW_INDIRECT_BUFFER);
}

void STPSkybox::initSkyboxRenderer(const STPShaderManager& skybox_fs, const STPSkyboxInitialiser& skybox_init) {
	if (skybox_fs.Type != GL_FRAGMENT_SHADER) {
		throw STPException::STPInvalidArgument("The shader initialised for skybox rendering must a fragment shader");
	}
	const auto& [skybox_vs, skybox_buf] = skybox_init;

	//construction of shared pointer throws exception if weak pointer is empty
	this->SkyboxVertex = shared_ptr(skybox_buf);

	this->SkyboxRenderer
		.attach(**skybox_vs)
		.attach(skybox_fs)
		.finalise();
}

void STPSkybox::drawSkybox() const {
	//bind
	this->SkyboxVertex->bind();
	this->SkyboxRenderer.use();

	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);

	//clean up
	STPProgramManager::unuse();
}