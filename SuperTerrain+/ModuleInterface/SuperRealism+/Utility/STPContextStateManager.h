#pragma once
#ifndef _STP_CONTEXT_STATE_MANAGER_H_
#define _STP_CONTEXT_STATE_MANAGER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Compaibility
#include <SuperTerrain+/STPOpenGL.h>

#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPContextStateManager is a GL state manager that records the current that and desired states,
	 * and only updates state whenever necessary.
	*/
	namespace STPContextStateManager {

		/**
		 * @brief STPContextState records all GL context states.
		 * Documentations all specify the initial values of the states when GL context is setup.
		*/
		struct STP_REALISM_API STPContextState {
		private:

			/**
			 * @brief STPBasicState contains status that shares across to different capabilities.
			*/
			struct STPBasicState {
			public:

				//Indication of whether a capability is turned on.
				bool Switch;

			};

		public:

			/**
			 * @brief The Depth Test is a per-sample processing operation performed after the Fragment Shader (and sometimes before). 
			 * The Fragment's output depth value may be tested against the depth of the sample being written to. 
			 * If the test fails, the fragment is discarded. 
			 * If the test passes, the depth buffer will be updated with the fragment's output depth, 
			 * unless a subsequent per-sample operation prevents it (such as turning off depth writes). 
			*/
			struct : public STPBasicState {
			public:

				//Default disabled

				//Depth test
				//Specify the value used for depth buffer comparisons.
				//GL_LESS
				STPOpenGL::STPenum DepthFunc;
				//Enable or disable writing into the depth buffer.
				//GL_TRUE
				STPOpenGL::STPboolean DepthMask;

			} DepthBuffer;

			/**
			 * @brief Triangle primitives after all transformation steps have a particular facing. 
			 * This is defined by the order of the three vertices that make up the triangle, as well as their apparent order on-screen. 
			 * Triangles can be discarded based on their apparent facing, a process known as Face Culling.
			*/
			struct : public STPBasicState {
			public:

				//Default disabled

				//Face culling
				//Define front- and back-facing polygons.
				//GL_CCW
				STPOpenGL::STPenum FrontFace;
				//Specify whether front- or back-facing facets can be culled.
				//GL_BACK
				STPOpenGL::STPenum CullFace;

			} CullFace;

			/**
			 * @brief Blending is the stage of OpenGL rendering pipeline that takes the fragment color outputs from the Fragment Shader 
			 * and combines them with the colors in the color buffers that these outputs map to. 
			 * Blending parameters can allow the source and destination colors for each output to be combined in various ways. 
			*/
			struct : public STPBasicState {
			public:

				//Default disabled

				//Blending
				//Specify pixel arithmetic, i.e., blend function.
				//GL_ONE, GL_ZERO
				STPOpenGL::STPenum FuncSource, FuncDestination;
				//Specify the equation used for both the RGB blend equation and the Alpha blend equation.
				//GL_FUNC_ADD
				STPOpenGL::STPenum Equation;

				//Set the blend color.
				//(0, 0, 0, 0)
				glm::vec4 BlendColor;

			} Blend;

			/**
			 * @brief Initialse a default GL context state.
			 * Default states follow against GL specification.
			*/
			STPContextState();

			~STPContextState() = default;

		};

		/**
		 * @brief STPCurrentState stores the current state of the GL context.
		 * This is a singleton class.
		*/
		class STP_REALISM_API STPCurrentState : private STPContextState {
		private:

			/**
			 * @brief Init the initial GL state.
			*/
			STPCurrentState() = default;

			~STPCurrentState() = default;

			/**
			 * @brief Change a boolean status only when the current status is different from the target.
			 * @param current_status The current status.
			 * @param target_status The new status to change to.
			 * @param type The type of this status to be updated to GL context.
			*/
			static void changeBinStatus(bool&, bool, STPOpenGL::STPenum);

			/**
			 * @brief Change a value status when the current status is different.
			 * @tparam V The type of the status value.
			 * @tparam Fun The GL function used to update the context.
			 * @param current_status The current status.
			 * @param target_status The new status to change to.
			 * @param gl_func The GL function to update the status.
			*/
			template<typename V, typename Fun>
			static void changeValueStatus(V&, V, Fun&&);

		public:

			STPCurrentState(const STPCurrentState&) = delete;

			STPCurrentState(STPCurrentState&&) = delete;

			STPCurrentState& operator=(const STPCurrentState&) = delete;

			STPCurrentState& operator=(STPCurrentState&&) = delete;

			/**
			 * @brief Get the current GL state instance.
			 * @return A pointer to the current GL state instance.
			*/
			static STPCurrentState& instance();

			/**
			 * @brief Get the state record of this GL state instance.
			 * @return A pointer to the current state.
			*/
			const STPContextState& currentState() const;

			/**
			 * @brief Set depth test status.
			 * @param status The new depth test status to be updated to GL context.
			*/
			void depthTest(bool);

			/**
			 * @brief Set the depth function.
			 * @param func The new depth function for GL context.
			*/
			void depthFunc(STPOpenGL::STPenum);

			/**
			 * @brief Set the depth mask.
			 * @param mask The new depth mask for GL context.
			*/
			void depthMask(STPOpenGL::STPboolean);

			/**
			 * @brief Set cull face status.
			 * @param status The new face culling status to be updated to GL context.
			*/
			void cullFace(bool);

			/**
			 * @brief Change the definition to front- and back-facing polygons.
			 * @param mode The new orientation of front facing polygons.
			*/
			void frontFace(STPOpenGL::STPenum);

			/**
			 * @brief Change the specification of whether front- or back-facing facets can be culled.
			 * @param mode The new face mode.
			*/
			void faceCulled(STPOpenGL::STPenum);

			/**
			 * @brief Set the blending status.
			 * @param sattus The new blending status to turn on or off.
			*/
			void blend(bool);

			/**
			 * @brief Specify the new blend function for pixel arithmetics.
			 * @param sfactor Specifies how the red, green, blue, and alpha source blending factors are computed.
			 * @param dfactor Specifies how the red, green, blue, and alpha destination blending factors are computed.
			*/
			void blendFunc(STPOpenGL::STPenum, STPOpenGL::STPenum);

			/**
			 * @brief Sprcify the new blend equation.
			 * @param mode Specifies how source and destination colors are combined.
			*/
			void blendEquation(STPOpenGL::STPenum);

			/**
			 * @brief Set the new blend color.
			 * @param color The new color for blending.
			*/
			void blendColor(glm::vec4);

		};

	};

}
#endif//_STP_CONTEXT_STATE_MANAGER_H_