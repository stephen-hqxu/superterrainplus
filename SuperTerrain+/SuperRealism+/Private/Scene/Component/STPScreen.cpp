#include <SuperRealism+/Scene/Component/STPScreen.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPValidationFailed.h>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

//GLAD
#include <glad/glad.h>

//System
#include <array>

using std::array;
using std::shared_ptr;

using glm::uvec2;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto ScreenShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPScreen", ".vert");

constexpr static array<signed char, 16u> QuadVertex = {
	//Position		//TexCoord
	-1, +1,			0, 1,
	-1, -1,			0, 0,
	+1, -1,			1, 0,
	+1, +1,			1, 1
};
constexpr static STPIndirectCommand::STPDrawArray QuadDrawCommand = {
	4u,
	1u,
	0u,
	0u
};

STPScreen::STPSimpleScreenFrameBuffer::STPSimpleScreenFrameBuffer() noexcept : ScreenColor(GL_TEXTURE_2D) {

}

STPTexture STPScreen::STPSimpleScreenFrameBuffer::updateScreenFrameBuffer(
	STPTexture* const stencil, const uvec2& dimension, const STPOpenGL::STPenum internal) {
	STP_ASSERTION_NUMERIC_DOMAIN(dimension.x > 0u && dimension.y > 0u, "Both component of a screen buffer dimension must be positive");
	//create new texture
	STPTexture new_screen_color(GL_TEXTURE_2D);
	//allocate memory
	new_screen_color.textureStorage2D(1, internal, dimension);

	//attach new texture to framebuffer
	this->ScreenColorContainer.attach(GL_COLOR_ATTACHMENT0, new_screen_color, 0);
	if (stencil) {
		this->ScreenColorContainer.attach(GL_STENCIL_ATTACHMENT, *stencil, 0);
	} else {
		//detach stencil buffer if none is provided because the framebuffer needs to be resized
		this->ScreenColorContainer.detachTexture(GL_STENCIL_ATTACHMENT);
	}
	//depth buffer is not needed because we are doing off-screen rendering
	this->ScreenColorContainer.validate(GL_FRAMEBUFFER);

	return new_screen_color;
}

void STPScreen::STPSimpleScreenFrameBuffer::setScreenBuffer(STPTexture* const stencil, const uvec2& dimension, const STPOpenGL::STPenum internal) {
	//store the new texture
	this->ScreenColor = std::move(this->updateScreenFrameBuffer(stencil, dimension, internal));
}

void STPScreen::STPSimpleScreenFrameBuffer::clearScreenBuffer(const vec4& color) noexcept {
	this->ScreenColorContainer.clearColor(0, color);
}

void STPScreen::STPSimpleScreenFrameBuffer::capture() const noexcept {
	this->ScreenColorContainer.bind(GL_FRAMEBUFFER);
}

STPScreen::STPSimpleScreenBindlessFrameBuffer::STPSimpleScreenBindlessFrameBuffer() noexcept : STPSimpleScreenFrameBuffer() {

}

void STPScreen::STPSimpleScreenBindlessFrameBuffer::setScreenBuffer(
	STPTexture* const stencil, const uvec2& dimension, const STPOpenGL::STPenum internal) {
	using std::move;
	STPTexture new_screen_color = this->updateScreenFrameBuffer(stencil, dimension, internal);
	//update bindless handle
	this->ScreenColorHandle = STPBindlessTexture::make(new_screen_color, this->ScreenColorSampler);
	this->ScreenColor = move(new_screen_color);
}

STPScreen::STPScreenVertexBuffer::STPScreenVertexBuffer() noexcept {
	//send of off screen quad
	this->ScreenBuffer.bufferStorageSubData(QuadVertex.data(), QuadVertex.size() * sizeof(signed char), GL_NONE);
	//rendering command
	this->ScreenRenderCommand.bufferStorageSubData(&QuadDrawCommand, sizeof(QuadDrawCommand), GL_NONE);

	//vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->ScreenArray.attribute();
	attr.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->ScreenBuffer, 0)
		.binding();
	this->ScreenArray.enable(0u, 2u);
}

void STPScreen::STPScreenVertexBuffer::bind() const noexcept {
	//bind vertex buffer
	this->ScreenArray.bind();
	this->ScreenRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
}

inline static STPShaderManager::STPShader createScreenVertexShader() {
	//read source
	const char* const screen_source_file = ScreenShaderFilename.data();
	STPShaderManager::STPShaderSource shader_source(screen_source_file, SuperTerrainPlus::STPFile::read(screen_source_file));
	return STPShaderManager::make(GL_VERTEX_SHADER, shader_source);
}

STPScreen::STPScreenInitialiser::STPScreenInitialiser() : VertexShader(createScreenVertexShader()) {
	
}

STPScreen::STPScreenProgramExecutor::STPScreenProgramExecutor(const STPScreen& screen) noexcept : OffScreenRendererState(screen.OffScreenRenderer.useManaged()) {
	screen.ScreenVertex->bind();
}

void STPScreen::STPScreenProgramExecutor::operator()() const noexcept {
	glDrawArraysIndirect(GL_TRIANGLE_FAN, nullptr);
}

void STPScreen::initScreenRenderer(const STPShaderManager::STPShader& screen_fs, const STPScreenInitialiser& screen_init) {
	STP_ASSERTION_VALIDATION(STPShaderManager::shaderType(screen_fs) == GL_FRAGMENT_SHADER,
		"The shader initialised for off-screen rendering must be a fragment shader");
	const auto& [screen_vs, screen_buf] = screen_init;

	//initialise screen vertex buffer
	this->ScreenVertex = shared_ptr(screen_buf);

	//setup screen program
	this->OffScreenRenderer = STPProgramManager({ &screen_vs, &screen_fs });
}

void STPScreen::drawScreen() const noexcept {
	const STPScreenProgramExecutor executor = this->drawScreenFromExecutor();
	executor();
}

STPScreen::STPScreenProgramExecutor STPScreen::drawScreenFromExecutor() const noexcept {
	return STPScreenProgramExecutor(*this);
}