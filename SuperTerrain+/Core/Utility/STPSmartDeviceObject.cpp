#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GL-CUDA
#include <glad/glad.h>
#include <cuda_gl_interop.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

using namespace SuperTerrainPlus;

/* STPStream */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPStreamDestroyer::operator()(const cudaStream_t stream) const {
	STP_CHECK_CUDA(cudaStreamDestroy(stream));
}

STPSmartDeviceObject::STPStream STPSmartDeviceObject::makeStream(const unsigned int flag) {
	cudaStream_t stream;
	STP_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flag));
	return STPStream(stream);
}

STPSmartDeviceObject::STPStream STPSmartDeviceObject::makeStream(const unsigned int flag, const int priority) {
	cudaStream_t stream;
	STP_CHECK_CUDA(cudaStreamCreateWithPriority(&stream, flag, priority));
	return STPStream(stream);
}

/* STPEvent */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPEventDestroyer::operator()(const cudaEvent_t event) const {
	STP_CHECK_CUDA(cudaEventDestroy(event));
}

STPSmartDeviceObject::STPEvent STPSmartDeviceObject::makeEvent(const unsigned int flag) {
	cudaEvent_t event;
	STP_CHECK_CUDA(cudaEventCreate(&event, flag));
	return STPEvent(event);
}

/* STPMemPool */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPMemPoolDestroyer::operator()(const cudaMemPool_t mem_pool) const {
	STP_CHECK_CUDA(cudaMemPoolDestroy(mem_pool));
}

STPSmartDeviceObject::STPMemPool STPSmartDeviceObject::makeMemPool(const cudaMemPoolProps& props) {
	cudaMemPool_t mem_pool;
	STP_CHECK_CUDA(cudaMemPoolCreate(&mem_pool, &props));
	return STPMemPool(mem_pool);
}

/* STPTexture */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPTextureDestroyer::operator()(const cudaTextureObject_t texture) const {
	STP_CHECK_CUDA(cudaDestroyTextureObject(texture));
}

STPSmartDeviceObject::STPTexture STPSmartDeviceObject::makeTexture(
	const cudaResourceDesc& resource_desc, const cudaTextureDesc& texture_desc, const cudaResourceViewDesc* const resview_desc) {
	cudaTextureObject_t texture;
	STP_CHECK_CUDA(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, resview_desc));
	return STPTexture(texture);
}

/* STPSurface */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPSurfaceDestroyer::operator()(const cudaSurfaceObject_t surface) const {
	STP_CHECK_CUDA(cudaDestroySurfaceObject(surface));
}

STPSmartDeviceObject::STPSurface STPSmartDeviceObject::makeSurface(const cudaResourceDesc& resource_desc) {
	cudaSurfaceObject_t surface;
	STP_CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
	return STPSurface(surface);
}

/* STPGraphicsResource */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGraphicsResourceUnregisterer::operator()(const cudaGraphicsResource_t resource) const {
	STP_CHECK_CUDA(cudaGraphicsUnregisterResource(resource));
}

STPSmartDeviceObject::STPGraphicsResource STPSmartDeviceObject::makeGLBufferResource(const STPOpenGL::STPuint buffer, const unsigned int flags) {
	cudaGraphicsResource_t resource;
	STP_CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resource, buffer, flags));
	return STPGraphicsResource(resource);
}

STPSmartDeviceObject::STPGraphicsResource STPSmartDeviceObject::makeGLImageResource(
	const STPOpenGL::STPuint image, const STPOpenGL::STPenum target, const unsigned int flags) {
	cudaGraphicsResource_t resource;
	STP_CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource, image, target, flags));
	return STPGraphicsResource(resource);
}

/* STPGLTextureObject */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGLTextureDeleter::operator()(const STPOpenGL::STPuint tbo) const noexcept {
	glDeleteTextures(1u, &tbo);
}

STPSmartDeviceObject::STPGLTextureObject STPSmartDeviceObject::makeGLTextureObject(const STPOpenGL::STPenum target) noexcept {
	GLuint tbo;
	glCreateTextures(target, 1u, &tbo);
	return STPGLTextureObject(tbo);
}

/* STPGLBindlessTextureHandle */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGLTextureHandleUnresidenter::operator()(const STPOpenGL::STPuint64 tHandle) const noexcept {
	glMakeTextureHandleNonResidentARB(tHandle);
}

STPSmartDeviceObject::STPGLBindlessTextureHandle STPSmartDeviceObject::makeGLBindlessTextureHandle(const STPOpenGL::STPuint texture) noexcept {
	const GLuint64 tHandle = glGetTextureHandleARB(texture);
	glMakeTextureHandleResidentARB(tHandle);

	return STPGLBindlessTextureHandle(tHandle);
}

STPSmartDeviceObject::STPGLBindlessTextureHandle STPSmartDeviceObject::makeGLBindlessTextureHandle(
	const STPOpenGL::STPuint texture, const STPOpenGL::STPuint sampler) noexcept {
	const GLuint64 tHandle = glGetTextureSamplerHandleARB(texture, sampler);
	glMakeTextureHandleResidentARB(tHandle);

	return STPGLBindlessTextureHandle(tHandle);
}