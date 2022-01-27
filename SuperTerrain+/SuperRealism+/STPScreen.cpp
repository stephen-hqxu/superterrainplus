#include <SuperRealism+/Renderer/STPScreen.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

//System
#include <array>

using std::array;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto ScreenShaderFilename =
	SuperTerrainPlus::STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPScreen", ".vert");

constexpr static array<signed char, 16ull> QuadVertex = {
	//Position		//TexCoord
	-1, +1,			0, 1,
	-1, -1,			0, 0,
	+1, -1,			1, 0,
	+1, +1,			1, 1
};
constexpr static array<unsigned char, 6ull> QuadIndex = {
	0, 1, 2,
	0, 2, 3
};
constexpr static STPIndirectCommand::STPDrawElement QuadDrawCommand = {
	static_cast<unsigned int>(QuadIndex.size()),
	1u,
	0u,
	0u,
	0u
};

STPScreen::STPScreen() {
	//send of off screen quad
	this->ScreenBuffer.bufferStorageSubData(QuadVertex.data(), QuadVertex.size() * sizeof(signed char), GL_NONE);
	this->ScreenIndex.bufferStorageSubData(QuadIndex.data(), QuadIndex.size() * sizeof(unsigned char), GL_NONE);
	//rendering command
	this->ScreenRenderCommand.bufferStorageSubData(&QuadDrawCommand, sizeof(QuadDrawCommand), GL_NONE);

	//vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->ScreenArray.attribute();
	attr.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->ScreenBuffer, 0)
		.elementBuffer(this->ScreenIndex)
		.binding();
	this->ScreenArray.enable(0u, 2u);
}

STPShaderManager STPScreen::compileScreenVertexShader(STPScreenLog& log) {
	STPShaderManager screen_shader(GL_VERTEX_SHADER);
	//read source
	STPShaderManager::STPShaderSource shader_source(*STPFile(ScreenShaderFilename.data()));
	//compile
	log.Log[0] = screen_shader(shader_source);

	return screen_shader;
}

void STPScreen::drawScreen() const {
	//bind vertex buffer
	this->ScreenArray.bind();
	this->ScreenRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);
}