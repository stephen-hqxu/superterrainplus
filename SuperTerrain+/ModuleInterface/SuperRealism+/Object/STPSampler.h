#pragma once
#ifndef _STP_SAMPLER_H_
#define _STP_SAMPLER_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Management
#include "STPNullableObject.hpp"

#include "STPImageParameter.hpp"

#include <limits>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSampler is a managed GL sampler object which stores sampling parameters for texture access inside a shader.
	*/
	class STP_REALISM_API STPSampler : public STPImageParameter {
	public:

		/**
		 * @brief STPSamplerUnbinder unbinds the sampler from the texture unit.
		*/
		struct STP_REALISM_API STPSamplerUnbinder {
		public:

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		//A smart texture unit state manager that automatically unbinds the sampler from specified texture unit to avoid state leakage.
		typedef STPUniqueResource<STPOpenGL::STPuint, std::numeric_limits<STPOpenGL::STPuint>::max(), STPSamplerUnbinder> STPSamplerUnitStateManager;

	private:

		/**
		 * @brief STPSamplerDeleter is an auto deleter for GL sampler.
		*/
		struct STP_REALISM_API STPSamplerDeleter {
		public:

			void operator()(STPOpenGL::STPuint) const noexcept;

		};
		typedef STPSmartGLuintObject<STPSamplerDeleter> STPSmartSampler;
		//SBO
		STPSmartSampler Sampler;

	public:

		/**
		 * @brief Create a new sampler object.
		*/
		STPSampler() noexcept;

		STPSampler(const STPSampler&) = delete;

		STPSampler(STPSampler&&) noexcept = default;

		STPSampler& operator=(const STPSampler&) = delete;

		STPSampler& operator=(STPSampler&&) noexcept = default;

		~STPSampler() = default;

		/**
		 * @brief Get the underlying sampler object.
		 * @return The GL sampler object.
		*/
		STPOpenGL::STPuint operator*() const noexcept;

		void filter(STPOpenGL::STPint, STPOpenGL::STPint) noexcept override;

		void wrap(STPOpenGL::STPint, STPOpenGL::STPint, STPOpenGL::STPint) noexcept override;

		void wrap(STPOpenGL::STPint) noexcept override;

		void borderColor(STPGLVector::STPfloatVec4) noexcept override;

		void borderColor(STPGLVector::STPintVec4) noexcept override;

		void borderColor(STPGLVector::STPuintVec4) noexcept override;

		void anisotropy(STPOpenGL::STPfloat) noexcept override;

		void compareFunction(STPOpenGL::STPint) noexcept override;

		void compareMode(STPOpenGL::STPint) noexcept override;

		/**
		 * @brief Bind a named sampler to a texturing target, with automatic binding state management.
		 * The motivation is sampler states override texture state, causing unwanted state leakage
		 * if one wishes to use built-in sampler from within the texture directly while the current texture unit has an active sampler.
		 * @param unit Specifies the index of the texture unit to which the sampler is bound.
		 * @return The sampler unit state manager.
		 * This manager will automatically unbind the sampler from this texture unit, based on std::unique_ptr.
		*/
		STPSamplerUnitStateManager bindManaged(STPOpenGL::STPuint) const noexcept;

	};

}
#endif//_STP_SAMPLER_H_