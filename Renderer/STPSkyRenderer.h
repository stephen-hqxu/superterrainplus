#pragma once

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief Render the skybox with day-night cycle
	*/
	class STPSkyRenderer {
	private:

		//drawing command
		const void* const command;
		//thread pool from the main drawing thread
		STPThreadPool* const rendering_pool;

		SglToolkit::SgTShaderProc skyShader;
		//buffers
		GLuint box_vbo, box_vao, box_ebo, box_indirect;
		GLuint tbo_sky;//cubemap array of 2
		//filenames to the textures
		//we will load the filename from ini when the constructor is called
		std::string path_day[6], path_night[6];
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
		//loading cubemaps in multi-thread
		std::future<STPTextureStorage*> Texloader_day[6], Texloader_night[6];

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
			GLchar* log = new GLchar[1024];

			this->skyShader.addShader(GL_VERTEX_SHADER, "./GLSL/STPsky.vert");
			this->skyShader.addShader(GL_FRAGMENT_SHADER, "./GLSL/STPsky.frag");
			if (this->skyShader.linkShader(log, 1024) != SglToolkit::SgTShaderProc::OK) {
				cerr << "\n---------------------STPSkyRenderer just crashed :(---------------------------" << endl;
				cerr << log << endl;
				cerr << "--------------------------------------------------------------------------------" << endl;
				//exit
				delete[] log;
				return false;
			}
			//texture
			glProgramUniform1i(this->skyShader.getP(), this->getLoc("SkyMap"), 0);

			//clear
			delete[] log;
			return true;
		}

		/**
		 * @brief Load the skybox cube model and textures
		*/
		void loadCube() {
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
			glNamedBufferStorage(this->box_vbo, sizeof(SglToolkit::SgTUtils::UNITBOX_VERTICES), SglToolkit::SgTUtils::UNITBOX_VERTICES, GL_NONE);
			glNamedBufferStorage(this->box_ebo, sizeof(SglToolkit::SgTUtils::UNITBOX_INDICES), SglToolkit::SgTUtils::UNITBOX_INDICES, GL_NONE);
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

			//the get() will auto-wait
			STPTextureStorage* reader = this->Texloader_day[0].get();//all textures must have the same size in cube map array so we choose an arbitary one
			//allocation
			glTextureStorage3D(this->tbo_sky, 1, GL_RGB8, reader->Width, reader->Height, 12);//6 faces * 2 layer
			for (int i = 0; i < 6; i++) {
				//day texture, stored in layer 0-5
				if (i != 0) {//we have already got index 0 at day time, we can only get() once
					reader = this->Texloader_day[i].get();
				}
				glTextureSubImage3D(this->tbo_sky, 0, 0, 0, i, reader->Width, reader->Height, 1, GL_RGB, GL_UNSIGNED_BYTE, reader->Texture);
				delete reader;//The class needs to be deleted

				//night texture stored in layer 6-11
				reader = this->Texloader_night[i].get();
				glTextureSubImage3D(this->tbo_sky, 0, 0, 0, i + 6, reader->Width, reader->Height, 1, GL_RGB, GL_UNSIGNED_BYTE, reader->Texture);
				delete reader;
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
		 * @param pool The thread pool for multi-threaded texture loading
		*/
		STPSkyRenderer(SIMPLE::SISection& day, SIMPLE::SISection& night, SIMPLE::SISection& globalPreset, DrawElementsIndirectCommand* const sky_cmd, STPThreadPool* const pool)
			: rotaingSpeed(std::stof(globalPreset("rotationSpeed"))), cyclingSpeed(std::stof(globalPreset("DaytimeSpeed"))), command(reinterpret_cast<void*>(sky_cmd)), rendering_pool(pool) {
			cout << "....Loading STPSkyRenderer....";
			
			//get the filename from the ini file
			for (int i = 0; i < 6; i++) {
				this->path_day[i] = "./Resource/" + day("folder") + "/" + day(this->LOADING_ORDER[i]) + "." + day("suffix");
				this->path_night[i] = "./Resource/" + night("folder") + "/" + night(this->LOADING_ORDER[i]) + "." + night("suffix");
			}
			//loading textures in multiple threads
			for (int i = 0; i < 6; i++) {
				this->Texloader_day[i] = this->rendering_pool->enqueue_future(STPTextureStorage::loadTexture, this->path_day[i].c_str(), 3);//we don't need alpha channel for skybox
				this->Texloader_night[i] = this->rendering_pool->enqueue_future(STPTextureStorage::loadTexture, this->path_night[i].c_str(), 3);
			}
			
			if (this->compileShader()) {
				cout << "Shader loaded :)" << endl;
			}
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
		void render(const double& frametime) {
			//timer for animations
			auto getRotations = [frametime](const float rotatingSpeed, float& rotations) -> mat4 {
				mat4 model = identity<mat4>();
				rotations += rotatingSpeed * static_cast<float>(frametime);
				//keeping the range [0,360], otherwise the float will become huge if the game keeps running
				if (rotations >= 360.0f) {
					rotations -= 360.0f;
				}

				return glm::rotate(model, radians(rotations), vec3(0.0f, 1.0f, 0.0f));
			};
			auto getDayNightFactor = [&frametime](const float cyclingSpeed, int& tick) -> float {
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
			//multithreading, why wasting our powerful CPU?
			static std::future<mat4> rotationCalc;
			static std::future<float> daynightCalc;
			rotationCalc = this->rendering_pool->enqueue_future(getRotations, this->rotaingSpeed, std::ref(this->rotations));
			daynightCalc = this->rendering_pool->enqueue_future(getDayNightFactor, this->cyclingSpeed, std::ref(this->tick));

			//sending uniforms
			glProgramUniformMatrix4fv(this->skyShader.getP(), this->getLoc("Rotations"), 1, GL_FALSE, value_ptr(rotationCalc.get()));
			glProgramUniform1f(this->skyShader.getP(), this->getLoc("factor"), daynightCalc.get());

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