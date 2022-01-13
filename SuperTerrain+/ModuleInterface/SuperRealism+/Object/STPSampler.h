#pragma once
#ifndef _STP_SAMPLER_H_
#define _STP_SAMPLER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Management
#include <SuperTerrain+/Utility/STPNullablePrimitive.h>

#include "STPImageParameter.hpp"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSampler is a managed GL sampler object which stores sampling parameters for texture acess inside a shader.
	*/
	class STP_REALISM_API STPSampler : private STPImageParameter {
	private:

		/**
		 * @brief STPSamplerDeleter is an auto deleter for GL sampler.
		*/
		struct STP_REALISM_API STPSamplerDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const;

		};
		typedef std::unique_ptr<STPOpenGL::STPuint, STPNullableGLuint::STPNullableDeleter<STPSamplerDeleter>> STPSmartSampler;
		//SBO
		STPSmartSampler Sampler;

	public:

		/**
		 * @brief Create a new sampler object.
		*/
		STPSampler();

		STPSampler(const STPSampler&) = delete;

		STPSampler(STPSampler&&) noexcept = default;

		STPSampler& operator=(const STPSampler&) = delete;

		STPSampler& operator=(STPSampler&&) noexcept = default;

		~STPSampler() = default;

		/**
		 * @brief Get the underlying sampler object.
		 * @return The GL sampler object.
		*/
		STPOpenGL::STPuint operator*() const;

		void filter(STPOpenGL::STPenum, STPOpenGL::STPenum) override;

		void wrap(STPOpenGL::STPenum, STPOpenGL::STPenum, STPOpenGL::STPenum) override;

		void wrap(STPOpenGL::STPenum) override;

		void borderColor(glm::vec4) override;

		void anisotropy(STPOpenGL::STPfloat) override;

		/**
		 * @brief Bind a named sampler to a texturing target.
		 * @param unit Specifies the index of the texture unit to which the sampler is bound.
		*/
		void bind(STPOpenGL::STPuint) const;

		/**
		 * @brief Unbind a texturing target from a sampler.
		 * @param unit Specifies the index of the texture unit to which the sampler is unbound.
		*/
		static void unbind(STPOpenGL::STPuint);

	};

}
#endif//_STP_SAMPLER_H_