#pragma once

//Parallel
#include <execution>
#include <algorithm>

namespace STPDemo {

	/**
	 * @brief Render the skybox with day-night cycle
	*/
	class STPSkyRenderer {
	private:

		//drawing command
		const void* const command;

		SglToolkit::SgTShaderProc skyShader;
		//buffers
		GLuint box_vbo, box_vao, box_ebo, box_indirect;
		GLuint tbo_sky;//cubemap array of 2
		//filenames to the textures
		//we will load the filename from ini when the constructor is called
		constexpr static size_t CubemapFaceCount = 12ull;
		//An index and the string filename
		std::pair<unsigned int, std::string> cubemap_path[STPSkyRenderer::CubemapFaceCount];
		const std::string LOADING_ORDER[6] = {
			"right",
			"left",
			"top",
			"bottom",
			"back",
			"front"
		};
		//sky animations settings
		const float rotaingSpeed, cyclingSpeed;
		float rotations = 0.0f;//in degree
		int tick = 0;

		/**
		 * @brief Get uniform location of the sky renderer
		 * @param name Name of the uniform
		 * @return The uniform location
		*/
		GLint getLoc(const GLchar* const name) {
			return glGetUniformLocation(this->skyShader.getP(), name);
		}

		/**
		 * @brief Compile the shader program for skybox
		 * @return The status of the compilation, true if successful
		*/
		const bool compileShader() {
			//error log
			GLchar log[1024];

			this->skyShader.addShader(GL_VERTEX_SHADER, "./GLSL/STPsky.vert");
			this->skyShader.addShader(GL_FRAGMENT_SHADER, "./GLSL/STPsky.frag");
			if (this->skyShader.linkShader(log, 1024) != SglToolkit::SgTShaderProc::OK) {
				cerr << "\n---------------------STPSkyRenderer just crashed :(---------------------------" << endl;
				cerr << log << endl;
				cerr << "--------------------------------------------------------------------------------" << endl;
				//exit
				return false;
			}
			//texture
			glProgramUniform1i(this->skyShader.getP(), this->getLoc("SkyMap"), 0);

			return true;
		}

		/**
		 * @brief Load the skybox cube model and textures
		*/
		void loadCube() {
			using glm::ivec3;

			//buffers
			glCreateBuffers(1, &this->box_vbo);
			glCreateBuffers(1, &this->box_indirect);
			glCreateVertexArrays(1, &this->box_vao);
			glCreateBuffers(1, &this->box_ebo);
			glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, 1, &this->tbo_sky);

			//creating indrect buffer
			glNamedBufferStorage(this->box_indirect, sizeof(DrawElementsIndirectCommand), this->command, GL_NONE);

			//loading the unit cube into vbo
			//set the flag to 0 to turn the buffer into fully-unmappable and immutable
			glNamedBufferStorage(this->box_vbo, sizeof(SglToolkit::SgTUtil::UNITBOX_VERTICES), SglToolkit::SgTUtil::UNITBOX_VERTICES, GL_NONE);
			glNamedBufferStorage(this->box_ebo, sizeof(SglToolkit::SgTUtil::UNITBOX_INDICES), SglToolkit::SgTUtil::UNITBOX_INDICES, GL_NONE);
			//assigning vao
			glVertexArrayVertexBuffer(this->box_vao, 0, this->box_vbo, 0, sizeof(int) * 3);
			glVertexArrayElementBuffer(this->box_vao, this->box_ebo);
			//attributing
			glEnableVertexArrayAttrib(this->box_vao, 0);
			glVertexArrayAttribFormat(this->box_vao, 0, 3, GL_INT, GL_FALSE, 0);
			glVertexArrayAttribBinding(this->box_vao, 0, 0);

			//loading texture
			glTextureParameteri(this->tbo_sky, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTextureParameteri(this->tbo_sky, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTextureParameteri(this->tbo_sky, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			glTextureParameteri(this->tbo_sky, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTextureParameteri(this->tbo_sky, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			//load all texture first
			STPTextureStorage texture[STPSkyRenderer::CubemapFaceCount];
			std::for_each_n(std::execution::par, this->cubemap_path, STPSkyRenderer::CubemapFaceCount, [&texture](const auto& filename) {
				//load texture from file, all textures are loading in the same order as the filename
				//we don't need alpha channel for skybox
				texture[filename.first] = STPTextureStorage(filename.second, 3);
			});
			{
				const ivec3 props = texture[0].property();
				//allocate storage in the first iteration
				//all textures must have the same size in cube map array so we choose an arbitary one
				glTextureStorage3D(this->tbo_sky, 1, GL_RGB8, props.x, props.y, STPSkyRenderer::CubemapFaceCount);//6 faces * 2 layer
			}
			for (unsigned int i = 0u; i < STPSkyRenderer::CubemapFaceCount; i++) {
				const STPTextureStorage& currTexture = texture[i];
				const ivec3& props = currTexture.property();
				glTextureSubImage3D(this->tbo_sky, 0, 0, 0, i, props.x, props.y, 1, GL_RGB, GL_UNSIGNED_BYTE, currTexture.texture());
			}
		}

		/**
		 * @brief Delete buffers and clearup the program
		*/
		void clearup() {
			this->skyShader.deleteShader();
			glDeleteBuffers(1, &this->box_vbo);
			glDeleteBuffers(1, &this->box_indirect);
			glDeleteVertexArrays(1, &this->box_vao);
			glDeleteBuffers(1, &box_ebo);
			glDeleteTextures(1, &tbo_sky);
		}

	public:

		/**
		 * @brief Init the sky renderer, providing the cubemap texture
		 * @param day The cubemap for daytime
		 * @param night The cubemap for nighttime
		 * @param globalPreset Glbal parameters
		 * @param sky_cmd The indrect rendering command for sky renderer
		*/
		STPSkyRenderer(const SIMPLE::SISection& day, const SIMPLE::SISection& night, const SIMPLE::SISection& globalPreset, const DrawElementsIndirectCommand* sky_cmd)
			: rotaingSpeed(globalPreset("rotationSpeed").to<float>()), cyclingSpeed(globalPreset("DaytimeSpeed").to<float>()), command(reinterpret_cast<const void*>(sky_cmd)) {
			cout << "....Loading STPSkyRenderer....";
			
			//get the filename from the ini file
			const size_t halfFaceCount = STPSkyRenderer::CubemapFaceCount / 2u;
			for (unsigned int i = 0u; i < STPSkyRenderer::CubemapFaceCount; i++) {
				this->cubemap_path[i] = std::make_pair(i, (i < halfFaceCount) ?
					"./Resource/" + day("folder")() + "/" + day(this->LOADING_ORDER[i])() + "." + day("suffix")() :
					"./Resource/" + night("folder")() + "/" + night(this->LOADING_ORDER[i - halfFaceCount])() + "." + night("suffix")());
			}
			
			if (!this->compileShader()) {
				std::terminate();
			}
			cout << "Shader loaded :)" << endl;
			this->loadCube();
			cout << "....Done...." << endl;
		}

		~STPSkyRenderer() {
			this->clearup();
		}

		/**
		 * @brief Return the program of the sky renderer
		 * @return The program reference number
		*/
		const GLint getSkyProgram() {
			return this->skyShader.getP();
		}

		/**
		 * @brief Render the sky
		 * PV matrix are sent via SSBO
		 * @param frametime Time elapsed on each frame
		*/
		void render(double frametime) {
			//timer for animations
			auto getRotations = [frametime](float rotatingSpeed, float& rotations) -> mat4 {
				mat4 model = identity<mat4>();
				rotations += rotatingSpeed * static_cast<float>(frametime);
				//keeping the range [0,360], otherwise the float will become huge if the game keeps running
				if (rotations >= 360.0f) {
					rotations -= 360.0f;
				}

				return glm::rotate(model, radians(rotations), vec3(0.0f, 1.0f, 0.0f));
			};
			auto getDayNightFactor = [frametime](float cyclingSpeed, int& tick) -> float {
				float factor = 0.0f;
				if (tick <= 11000) {//day
					factor = glm::smoothstep<float>(8000.0f, 11000.0f, tick * 1.0f);
				}
				else {//11000-24000, night
					factor = 1.0f - glm::smoothstep<float>(21000.0f, 24000.0f, tick * 1.0f);
				}

				//accumulating the tick
				tick += static_cast<int>(frametime * cyclingSpeed);
				tick %= 24000;//24000 ticks per day, just like minecraft

				return factor;
			};
			mat4 rotationCalc;
			float daynightCalc;
			rotationCalc = getRotations(this->rotaingSpeed, this->rotations);
			daynightCalc = getDayNightFactor(this->cyclingSpeed, this->tick);

			//sending uniforms
			glProgramUniformMatrix4fv(this->skyShader.getP(), this->getLoc("Rotations"), 1, GL_FALSE, value_ptr(rotationCalc));
			glProgramUniform1f(this->skyShader.getP(), this->getLoc("factor"), daynightCalc);

			//render
			glBindTextureUnit(0, this->tbo_sky);
			glBindVertexArray(this->box_vao);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, this->box_indirect);
			glUseProgram(this->skyShader.getP());
			glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr);
			glUseProgram(0);
		}
	};
}