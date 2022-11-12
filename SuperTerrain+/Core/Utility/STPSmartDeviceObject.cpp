#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GL-CUDA
#include <glad/glad.h>
#include <cuda_gl_interop.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

using namespace SuperTerrainPlus;

/* STPStream */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPStreamDestroyer::operator()(cudaStream_t stream) const {
	STP_CHECK_CUDA(cudaStreamDestroy(stream));
}

STPSmartDeviceObject::STPStream STPSmartDeviceObject::makeStream(unsigned int flag) {
	cudaStream_t stream;
	STP_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, flag));
	return STPStream(stream);
}

STPSmartDeviceObject::STPStream STPSmartDeviceObject::makeStream(unsigned int flag, int priority) {
	cudaStream_t stream;
	STP_CHECK_CUDA(cudaStreamCreateWithPriority(&stream, flag, priority));
	return STPStream(stream);
}

/* STPEvent */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPEventDestroyer::operator()(cudaEvent_t event) const {
	STP_CHECK_CUDA(cudaEventDestroy(event));
}

STPSmartDeviceObject::STPEvent STPSmartDeviceObject::makeEvent(unsigned int flag) {
	cudaEvent_t event;
	STP_CHECK_CUDA(cudaEventCreate(&event, flag));
	return STPEvent(event);
}

/* STPMemPool */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPMemPoolDestroyer::operator()(cudaMemPool_t mem_pool) const {
	STP_CHECK_CUDA(cudaMemPoolDestroy(mem_pool));
}

STPSmartDeviceObject::STPMemPool STPSmartDeviceObject::makeMemPool(const cudaMemPoolProps& props) {
	cudaMemPool_t mem_pool;
	STP_CHECK_CUDA(cudaMemPoolCreate(&mem_pool, &props));
	return STPMemPool(mem_pool);
}

/* STPTexture */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPTextureDestroyer::operator()(cudaTextureObject_t texture) const {
	STP_CHECK_CUDA(cudaDestroyTextureObject(texture));
}

STPSmartDeviceObject::STPTexture STPSmartDeviceObject::makeTexture(
	const cudaResourceDesc& resource_desc, const cudaTextureDesc& texture_desc, const cudaResourceViewDesc* resview_desc) {
	cudaTextureObject_t texture;
	STP_CHECK_CUDA(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, resview_desc));
	return STPTexture(texture);
}

/* STPSurface */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPSurfaceDestroyer::operator()(cudaSurfaceObject_t surface) const {
	STP_CHECK_CUDA(cudaDestroySurfaceObject(surface));
}

STPSmartDeviceObject::STPSurface STPSmartDeviceObject::makeSurface(const cudaResourceDesc& resource_desc) {
	cudaSurfaceObject_t surface;
	STP_CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resource_desc));
	return STPSurface(surface);
}

/* STPGraphicsResource */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGraphicsResourceUnregisterer::operator()(cudaGraphicsResource_t resource) const {
	STP_CHECK_CUDA(cudaGraphicsUnregisterResource(resource));
}

STPSmartDeviceObject::STPGraphicsResource STPSmartDeviceObject::makeGLBufferResource(STPOpenGL::STPuint buffer, unsigned int flags) {
	cudaGraphicsResource_t resource;
	STP_CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resource, buffer, flags));
	return STPGraphicsResource(resource);
}

STPSmartDeviceObject::STPGraphicsResource STPSmartDeviceObject::makeGLImageResource(STPOpenGL::STPuint image, STPOpenGL::STPenum target, unsigned int flags) {
	cudaGraphicsResource_t resource;
	STP_CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource, image, target, flags));
	return STPGraphicsResource(resource);
}

/* STPGLTextureObject */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGLTextureDeleter::operator()(STPOpenGL::STPuint tbo) const noexcept {
	glDeleteTextures(1u, &tbo);
}

STPSmartDeviceObject::STPGLTextureObject STPSmartDeviceObject::makeGLTextureObject(STPOpenGL::STPenum target) noexcept {
	GLuint tbo;
	glCreateTextures(target, 1u, &tbo);
	return STPGLTextureObject(tbo);
}

/* STPGLBindlessTextureHandle */

void STPSmartDeviceObject::STPSmartDeviceObjectImpl::STPGLTextureHandleUnresidenter::operator()(STPOpenGL::STPuint64 tHandle) const noexcept {
	glMakeTextureHandleNonResidentARB(tHandle);
}

STPSmartDeviceObject::STPGLBindlessTextureHandle STPSmartDeviceObject::makeGLBindlessTextureHandle(STPOpenGL::STPuint texture) noexcept {
	const GLuint64 tHandle = glGetTextureHandleARB(texture);
	glMakeTextureHandleResidentARB(tHandle);

	return STPGLBindlessTextureHandle(tHandle);
}

STPSmartDeviceObject::STPGLBindlessTextureHandle STPSmartDeviceObject::makeGLBindlessTextureHandle(STPOpenGL::STPuint texture, STPOpenGL::STPuint sampler) noexcept {
	const GLuint64 tHandle = glGetTextureSamplerHandleARB(texture, sampler);
	glMakeTextureHandleResidentARB(tHandle);

	return STPGLBindlessTextureHandle(tHandle);
}