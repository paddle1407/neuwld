/* wld: gbm.c
 *
 * A GBM/EGL/GLES2 accelerated backend.
 *
 * Unlike intel.c and nouveau.c, which talk to specific hardware through
 * libdrm_intel / libdrm_nouveau, this backend goes through the generic
 * GBM + EGL stack. That means it works on any driver providing a GBM
 * backend, including the proprietary NVIDIA driver, which has no
 * libdrm_* of its own.
 *
 * Copyright (c) 2026 the neuwld authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "drm-private.h"
#include "drm.h"
#include "wld-private.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <drm_fourcc.h>
#include <gbm.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include <fontconfig/fontconfig.h>

/* Number of glyph textures cached per renderer. */
#define GLYPH_CACHE_SIZE 512

struct glyph_entry {
	struct glyph *glyph;
	GLuint texture;
	uint32_t width, height;
};

struct gbm_context {
	struct wld_context base;
	int fd;
	struct gbm_device *gbm;
	EGLDisplay display;
	EGLContext context;

	PFNEGLCREATEIMAGEKHRPROC create_image;
	PFNEGLDESTROYIMAGEKHRPROC destroy_image;
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture_2d;
	PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_dmabuf_modifiers;

	bool has_modifiers;
	bool has_unpack_subimage;
};

struct gbm_buffer {
	struct buffer base;
	struct wld_exporter exporter;
	struct gbm_context *context;

	/* NULL for buffers imported from a client's dmabuf. */
	struct gbm_bo *bo;
	EGLImageKHR image;
	GLuint texture;
	uint32_t handle;
	bool own_handle;
	void *map_data;

	/*
	 * WLD_FLAG_MAP buffers are backed by a DRM dumb buffer rather than GBM.
	 * NVIDIA will not sample or render a linear dmabuf, so these are uploaded
	 * into an ordinary GL texture instead of being wrapped in an EGLImage.
	 */
	bool cpu;
	bool dumb;
	bool tex_allocated;
	size_t mapping_size;
	bool dirty;
};

struct gles_renderer {
	struct wld_renderer base;
	struct gbm_context *context;

	GLuint fbo;
	GLuint target_texture;
	uint32_t target_width, target_height;

	struct {
		GLuint program;
		GLint pos, proj, color;
	} solid;
	struct {
		GLuint program;
		GLint pos, texcoord, proj, tex, mul, add;
	} textured;
	struct {
		GLuint program;
		GLint pos, texcoord, proj, tex, color;
	} glyph;

	GLfloat proj[16];
	struct glyph_entry glyphs[GLYPH_CACHE_SIZE];
};

#define CONTEXT_IMPLEMENTS_QUERY_MODIFIERS
#define BUFFER_IMPLEMENTS_FLUSH
#define RENDERER_IMPLEMENTS_REGION
#define RENDERER_IMPLEMENTS_BLEND
#define RENDERER_IMPLEMENTS_READ_PIXELS
#include "interface/buffer.h"
#include "interface/context.h"
#include "interface/renderer.h"
#define DRM_DRIVER_NAME gbm
#include "interface/drm.h"
IMPL(gbm_context, wld_context)
IMPL(gles_renderer, wld_renderer)
IMPL(gbm_buffer, wld_buffer)

/**** Shaders ****/

static const char vertex_solid_src[] =
    "uniform mat4 proj;\n"
    "attribute vec2 pos;\n"
    "void main() { gl_Position = proj * vec4(pos, 0.0, 1.0); }\n";

static const char fragment_solid_src[] =
    "precision mediump float;\n"
    "uniform vec4 color;\n"
    "void main() { gl_FragColor = color; }\n";

static const char vertex_tex_src[] =
    "uniform mat4 proj;\n"
    "attribute vec2 pos;\n"
    "attribute vec2 texcoord;\n"
    "varying vec2 v_tex;\n"
    "void main() {\n"
    "  v_tex = texcoord;\n"
    "  gl_Position = proj * vec4(pos, 0.0, 1.0);\n"
    "}\n";

/* mul/add let an XRGB source be forced opaque without a second shader. */
static const char fragment_tex_src[] =
    "precision mediump float;\n"
    "varying vec2 v_tex;\n"
    "uniform sampler2D tex;\n"
    "uniform vec4 mul;\n"
    "uniform vec4 add;\n"
    "void main() { gl_FragColor = texture2D(tex, v_tex) * mul + add; }\n";

static const char fragment_glyph_src[] =
    "precision mediump float;\n"
    "varying vec2 v_tex;\n"
    "uniform sampler2D tex;\n"
    "uniform vec4 color;\n"
    "void main() { gl_FragColor = color * texture2D(tex, v_tex).a; }\n";

static GLuint
compile_shader(GLenum type, const char *source)
{
	GLuint shader;
	GLint status;

	shader = glCreateShader(type);
	if (!shader)
		return 0;

	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	if (status == GL_FALSE) {
		char log[512];
		glGetShaderInfoLog(shader, sizeof log, NULL, log);
		DEBUG("shader compilation failed: %s\n", log);
		glDeleteShader(shader);
		return 0;
	}

	return shader;
}

static GLuint
link_program(const char *vertex_source, const char *fragment_source)
{
	GLuint vertex, fragment, program;
	GLint status;

	if (!(vertex = compile_shader(GL_VERTEX_SHADER, vertex_source)))
		return 0;
	if (!(fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source))) {
		glDeleteShader(vertex);
		return 0;
	}

	program = glCreateProgram();
	glAttachShader(program, vertex);
	glAttachShader(program, fragment);
	glLinkProgram(program);
	glDeleteShader(vertex);
	glDeleteShader(fragment);

	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		char log[512];
		glGetProgramInfoLog(program, sizeof log, NULL, log);
		DEBUG("program link failed: %s\n", log);
		glDeleteProgram(program);
		return 0;
	}

	return program;
}

/**** Colour helpers ****/

/*
 * wld colours are non-premultiplied 0xAARRGGBB. Everything downstream of us
 * blends with premultiplied alpha, so premultiply on the way in.
 */
static void
color_to_gl(uint32_t color, GLfloat out[4])
{
	GLfloat a = ((color >> 24) & 0xff) / 255.0f;

	out[0] = (((color >> 16) & 0xff) / 255.0f) * a;
	out[1] = (((color >> 8) & 0xff) / 255.0f) * a;
	out[2] = ((color & 0xff) / 255.0f) * a;
	out[3] = a;
}

/**** Driver ****/

bool
driver_device_supported(uint32_t vendor_id, uint32_t device_id)
{
	/*
	 * GBM is driver agnostic, so we cannot decide from the PCI ID alone.
	 * Claim everything and let driver_create_context() fail if the stack
	 * is not actually usable; drm.c then falls through to the dumb driver.
	 */
	return !getenv("WLD_DRM_NO_GBM");
}

static bool
has_extension(const char *extensions, const char *extension)
{
	size_t length = strlen(extension);
	const char *p = extensions;

	if (!extensions)
		return false;

	while ((p = strstr(p, extension))) {
		if ((p == extensions || p[-1] == ' ')
		    && (p[length] == ' ' || p[length] == '\0')) {
			return true;
		}
		p += length;
	}

	return false;
}

struct wld_context *
driver_create_context(int drm_fd)
{
	struct gbm_context *context;
	const char *extensions;
	EGLint major, minor;
	EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};
	EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};
	EGLConfig config = EGL_NO_CONFIG_KHR;
	EGLint num_configs;

	if (!(context = malloc(sizeof *context)))
		return NULL;

	context->fd = drm_fd;

	if (!(context->gbm = gbm_create_device(drm_fd))) {
		DEBUG("gbm_create_device failed\n");
		goto error0;
	}

	context->display = eglGetDisplay((EGLNativeDisplayType)context->gbm);
	if (context->display == EGL_NO_DISPLAY) {
		DEBUG("eglGetDisplay failed\n");
		goto error1;
	}

	if (!eglInitialize(context->display, &major, &minor)) {
		DEBUG("eglInitialize failed\n");
		goto error1;
	}

	extensions = eglQueryString(context->display, EGL_EXTENSIONS);
	if (!has_extension(extensions, "EGL_KHR_image_base")
	    || !has_extension(extensions, "EGL_EXT_image_dma_buf_import")) {
		DEBUG("required EGL image extensions missing\n");
		goto error2;
	}
	context->has_modifiers =
	    has_extension(extensions, "EGL_EXT_image_dma_buf_import_modifiers");

	if (!eglBindAPI(EGL_OPENGL_ES_API)) {
		DEBUG("eglBindAPI failed\n");
		goto error2;
	}

	/*
	 * We only ever render to FBOs backed by EGLImages, never to an EGL
	 * surface, so a config is unnecessary and a surfaceless context is
	 * enough.
	 */
	if (!has_extension(extensions, "EGL_KHR_surfaceless_context")) {
		DEBUG("EGL_KHR_surfaceless_context missing\n");
		goto error2;
	}

	/*
	 * We render only into FBOs backed by EGLImages, so the config is never
	 * used. Prefer a config-less context where available and otherwise take
	 * whatever eglChooseConfig hands back.
	 */
	if (!has_extension(extensions, "EGL_KHR_no_config_context")) {
		if (!eglChooseConfig(context->display, config_attribs, &config, 1,
		                     &num_configs)
		    || num_configs == 0) {
			DEBUG("eglChooseConfig found no usable config\n");
			goto error2;
		}
	}

	context->context = eglCreateContext(context->display, config,
	                                    EGL_NO_CONTEXT, context_attribs);
	if (context->context == EGL_NO_CONTEXT) {
		DEBUG("eglCreateContext failed\n");
		goto error2;
	}

	if (!eglMakeCurrent(context->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
	                    context->context)) {
		DEBUG("eglMakeCurrent failed\n");
		goto error3;
	}

	context->create_image = (PFNEGLCREATEIMAGEKHRPROC)
	    eglGetProcAddress("eglCreateImageKHR");
	context->destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)
	    eglGetProcAddress("eglDestroyImageKHR");
	context->image_target_texture_2d = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
	    eglGetProcAddress("glEGLImageTargetTexture2DOES");
	context->query_dmabuf_modifiers = (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)
	    eglGetProcAddress("eglQueryDmaBufModifiersEXT");

	if (!context->create_image || !context->destroy_image
	    || !context->image_target_texture_2d) {
		DEBUG("required EGL/GL entry points missing\n");
		goto error3;
	}

	{
		const char *gl_ext = (const char *)glGetString(GL_EXTENSIONS);
		context->has_unpack_subimage =
		    has_extension(gl_ext, "GL_EXT_unpack_subimage");
	}

	context_initialize(&context->base, &wld_context_impl);
	DEBUG("using GBM/EGL context (EGL %d.%d)\n", major, minor);

	return &context->base;

error3:
	eglDestroyContext(context->display, context->context);
error2:
	eglTerminate(context->display);
error1:
	gbm_device_destroy(context->gbm);
error0:
	free(context);
	return NULL;
}

/**** Buffer ****/

static bool
export(struct wld_exporter *exporter, struct wld_buffer *base,
       uint32_t type, union wld_object *object)
{
	struct gbm_buffer *buffer = gbm_buffer(base);

	switch (type) {
	case WLD_DRM_OBJECT_HANDLE:
		if (!buffer->handle)
			return false;
		object->u32 = buffer->handle;
		return true;
	case WLD_DRM_OBJECT_MODIFIER:
		/* Dumb buffers are always linear. */
		object->u64 = buffer->bo ? gbm_bo_get_modifier(buffer->bo)
		                         : DRM_FORMAT_MOD_LINEAR;
		return true;
	case WLD_DRM_OBJECT_PRIME_FD:
		if (buffer->bo) {
			object->i = gbm_bo_get_fd(buffer->bo);
			return object->i >= 0;
		}
		if (drmPrimeHandleToFD(buffer->context->fd, buffer->handle,
		                       DRM_CLOEXEC, &object->i)
		    != 0) {
			return false;
		}
		return true;
	default:
		return false;
	}
}

/*
 * Build an EGLImage for a single-plane dmabuf. Modifiers are passed only when
 * the driver advertises support and the modifier is not INVALID, since some
 * drivers reject the modifier attributes outright.
 */
static EGLImageKHR
image_from_dmabuf(struct gbm_context *context, int fd, uint32_t width,
                  uint32_t height, uint32_t format, uint32_t pitch,
                  uint32_t offset, uint64_t modifier)
{
	EGLint attribs[30];
	size_t n = 0;

	attribs[n++] = EGL_WIDTH;
	attribs[n++] = width;
	attribs[n++] = EGL_HEIGHT;
	attribs[n++] = height;
	attribs[n++] = EGL_LINUX_DRM_FOURCC_EXT;
	attribs[n++] = format;
	attribs[n++] = EGL_DMA_BUF_PLANE0_FD_EXT;
	attribs[n++] = fd;
	attribs[n++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT;
	attribs[n++] = offset;
	attribs[n++] = EGL_DMA_BUF_PLANE0_PITCH_EXT;
	attribs[n++] = pitch;

	if (context->has_modifiers && modifier != DRM_FORMAT_MOD_INVALID) {
		attribs[n++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
		attribs[n++] = modifier & 0xffffffff;
		attribs[n++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
		attribs[n++] = modifier >> 32;
	}

	attribs[n++] = EGL_NONE;

	return context->create_image(context->display, EGL_NO_CONTEXT,
	                             EGL_LINUX_DMA_BUF_EXT, NULL, attribs);
}

static struct buffer *
new_buffer(struct gbm_context *context, struct gbm_bo *bo, EGLImageKHR image,
           uint32_t handle, bool own_handle, uint32_t width, uint32_t height,
           uint32_t format, uint32_t pitch)
{
	struct gbm_buffer *buffer;

	if (!(buffer = calloc(1, sizeof *buffer)))
		return NULL;

	buffer_initialize(&buffer->base, &wld_buffer_impl, width, height, format,
	                  pitch);
	buffer->context = context;
	buffer->bo = bo;
	buffer->image = image;
	buffer->texture = 0;
	buffer->handle = handle;
	buffer->own_handle = own_handle;
	buffer->map_data = NULL;
	buffer->exporter.export = &export;
	wld_buffer_add_exporter(&buffer->base.base, &buffer->exporter);

	return &buffer->base;
}

struct buffer *
context_create_buffer(struct wld_context *base, uint32_t width, uint32_t height,
                      uint32_t format, uint32_t flags)
{
	struct gbm_context *context = gbm_context(base);
	struct gbm_bo *bo;
	EGLImageKHR image;
	struct buffer *buffer;
	uint32_t usage = GBM_BO_USE_RENDERING;
	int fd;

	/*
	 * A buffer the CPU has to write cannot be a GBM buffer here. NVIDIA
	 * refuses GBM_BO_USE_LINEAR alongside RENDERING, and a buffer created
	 * with the LINEAR modifier can be neither rendered to nor sampled --
	 * sampling one fails the draw with GL_INVALID_OPERATION.
	 *
	 * A DRM dumb buffer gives us what these are actually used for: CPU
	 * writes, and a GEM handle for the cursor plane. When one is used as a
	 * drawing source we upload it into a normal GL texture, the same way a
	 * compositor handles a client's shm buffer.
	 */
	if (flags & WLD_FLAG_MAP) {
		struct gbm_buffer *cpu_buffer;
		uint32_t pitch = width * format_bytes_per_pixel(format);
		void *data;

		/*
		 * Only the cursor plane needs a GEM handle, and a dumb buffer's
		 * mapping is write-combined: reading 3MB back out of one to upload
		 * it takes ~190ms, which is far slower than compositing. Everything
		 * else gets ordinary cached memory, which the GPU reads at memcpy
		 * speed. Cursors stay dumb and are small enough not to care.
		 */
		if (flags & WLD_FLAG_CURSOR) {
			struct drm_mode_create_dumb create_dumb = {
				.height = height,
				.width = width,
				.bpp = format_bytes_per_pixel(format) * 8,
			};
			struct drm_mode_map_dumb map_dumb;

			if (drmIoctl(context->fd, DRM_IOCTL_MODE_CREATE_DUMB,
			             &create_dumb) != 0) {
				DEBUG("DRM_IOCTL_MODE_CREATE_DUMB failed\n");
				return NULL;
			}

			map_dumb = (struct drm_mode_map_dumb){ .handle = create_dumb.handle };
			if (drmIoctl(context->fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb) != 0)
				goto error_dumb;

			data = mmap(NULL, create_dumb.size, PROT_READ | PROT_WRITE,
			            MAP_SHARED, context->fd, map_dumb.offset);
			if (data == MAP_FAILED)
				goto error_dumb;

			buffer = new_buffer(context, NULL, EGL_NO_IMAGE_KHR,
			                    create_dumb.handle, false, width, height,
			                    format, create_dumb.pitch);
			if (!buffer) {
				munmap(data, create_dumb.size);
				goto error_dumb;
			}

			cpu_buffer = gbm_buffer(&buffer->base);
			cpu_buffer->dumb = true;
			cpu_buffer->mapping_size = create_dumb.size;
			goto cpu_done;

		error_dumb: {
			struct drm_mode_destroy_dumb destroy_dumb = {
				.handle = create_dumb.handle
			};
			drmIoctl(context->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_dumb);
		}
			return NULL;
		}

		if (!(data = calloc(height, pitch)))
			return NULL;

		buffer = new_buffer(context, NULL, EGL_NO_IMAGE_KHR, 0, false, width,
		                    height, format, pitch);
		if (!buffer) {
			free(data);
			return NULL;
		}
		cpu_buffer = gbm_buffer(&buffer->base);

	cpu_done:
		cpu_buffer->cpu = true;
		cpu_buffer->dirty = true;
		buffer->base.map = data;
		return buffer;
	}

	if (flags & WLD_DRM_FLAG_SCANOUT)
		usage |= GBM_BO_USE_SCANOUT;

	if (!(bo = gbm_bo_create(context->gbm, width, height, format, usage))) {
		DEBUG("gbm_bo_create failed (%ux%u flags 0x%x)\n", width, height, flags);
		return NULL;
	}

	if ((fd = gbm_bo_get_fd(bo)) < 0)
		goto error0;

	image = image_from_dmabuf(context, fd, width, height, format,
	                          gbm_bo_get_stride(bo), gbm_bo_get_offset(bo, 0),
	                          gbm_bo_get_modifier(bo));
	close(fd);

	if (image == EGL_NO_IMAGE_KHR) {
		DEBUG("failed to create EGLImage for new buffer\n");
		goto error0;
	}

	buffer = new_buffer(context, bo, image, gbm_bo_get_handle(bo).u32, false,
	                    width, height, format, gbm_bo_get_stride(bo));
	if (!buffer)
		goto error1;

	return buffer;

error1:
	context->destroy_image(context->display, image);
error0:
	gbm_bo_destroy(bo);
	return NULL;
}

struct buffer *
context_import_buffer(struct wld_context *base, uint32_t type,
                      union wld_object object, uint32_t width, uint32_t height,
                      uint32_t format, uint32_t pitch)
{
	struct gbm_context *context = gbm_context(base);
	EGLImageKHR image;
	struct buffer *buffer;
	uint32_t handle;

	int fd;
	uint32_t offset = 0;
	uint64_t modifier = DRM_FORMAT_MOD_INVALID;

	switch (type) {
	case WLD_DRM_OBJECT_PRIME_FD:
		fd = object.i;
		break;
	case WLD_DRM_OBJECT_DMABUF: {
		const struct wld_dmabuf_attributes *attributes = object.ptr;

		fd = attributes->fd;
		offset = attributes->offset;
		pitch = attributes->pitch;
		modifier = attributes->modifier;
		break;
	}
	default:
		return NULL;
	}

	image = image_from_dmabuf(context, fd, width, height, format, pitch, offset,
	                          modifier);
	if (image == EGL_NO_IMAGE_KHR) {
		DEBUG("failed to import client dmabuf as EGLImage\n");
		return NULL;
	}

	/* Needed only if this buffer is ever scanned out directly. */
	if (drmPrimeFDToHandle(context->fd, fd, &handle) != 0)
		handle = 0;

	buffer = new_buffer(context, NULL, image, handle, handle != 0, width,
	                    height, format, pitch);
	if (!buffer) {
		context->destroy_image(context->display, image);
		return NULL;
	}

	return buffer;
}

int
context_query_modifiers(struct wld_context *base, uint32_t format,
                        uint64_t *modifiers, int max)
{
	struct gbm_context *context = gbm_context(base);
	EGLint count = 0;

	if (!context->query_dmabuf_modifiers || !context->has_modifiers)
		return -1;

	if (!context->query_dmabuf_modifiers(context->display, format, 0, NULL, NULL,
	                                     &count)) {
		return -1;
	}
	if (count > max)
		count = max;
	if (count <= 0)
		return 0;

	if (!context->query_dmabuf_modifiers(context->display, format, count,
	                                     (EGLuint64KHR *)modifiers, NULL,
	                                     &count)) {
		return -1;
	}

	return count;
}

void
context_destroy(struct wld_context *base)
{
	struct gbm_context *context = gbm_context(base);

	eglMakeCurrent(context->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
	               EGL_NO_CONTEXT);
	eglDestroyContext(context->display, context->context);
	eglTerminate(context->display);
	gbm_device_destroy(context->gbm);
	close(context->fd);
	free(context);
}

bool
buffer_map(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);
	uint32_t stride;
	void *data;

	/* Dumb buffers stay mapped for their lifetime. */
	if (buffer->cpu)
		return base->base.map != NULL;

	/* Imported client dmabufs are not CPU accessible through GBM. */
	if (!buffer->bo)
		return false;

	data = gbm_bo_map(buffer->bo, 0, 0, base->base.width, base->base.height,
	                  GBM_BO_TRANSFER_READ_WRITE, &stride, &buffer->map_data);
	if (!data || data == MAP_FAILED)
		return false;

	/*
	 * wld fixes the pitch at buffer creation, so a map that hands back a
	 * different stride cannot be represented.
	 */
	if (stride != base->base.pitch) {
		DEBUG("map stride %u != buffer pitch %u\n", stride, base->base.pitch);
		gbm_bo_unmap(buffer->bo, buffer->map_data);
		buffer->map_data = NULL;
		return false;
	}

	base->base.map = data;
	return true;
}

bool
buffer_unmap(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	if (buffer->cpu) {
		/* The CPU may have written; the texture is now stale. */
		buffer->dirty = true;
		return true;
	}

	if (!buffer->bo || !buffer->map_data)
		return false;

	gbm_bo_unmap(buffer->bo, buffer->map_data);
	buffer->map_data = NULL;
	base->base.map = NULL;

	return true;
}

void
buffer_flush(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	/*
	 * Called once the renderer targeting this buffer is done with it, which
	 * for a dumb buffer means the CPU has just finished drawing into it.
	 */
	if (buffer->cpu)
		buffer->dirty = true;
}

void
buffer_destroy(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	if (buffer->texture)
		glDeleteTextures(1, &buffer->texture);

	if (buffer->cpu) {
		if (buffer->dumb) {
			struct drm_mode_destroy_dumb destroy_dumb = {
				.handle = buffer->handle
			};

			if (base->base.map)
				munmap(base->base.map, buffer->mapping_size);
			drmIoctl(buffer->context->fd, DRM_IOCTL_MODE_DESTROY_DUMB,
			         &destroy_dumb);
		} else {
			free(base->base.map);
		}
		free(buffer);
		return;
	}

	if (buffer->image != EGL_NO_IMAGE_KHR) {
		buffer->context->destroy_image(buffer->context->display,
		                               buffer->image);
	}
	if (buffer->bo)
		gbm_bo_destroy(buffer->bo);
	else if (buffer->own_handle) {
		struct drm_gem_close close_arg = { .handle = buffer->handle };
		drmIoctl(buffer->context->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
	}

	free(buffer);
}

/**** Renderer ****/

/* Lazily wrap a buffer's EGLImage in a GL texture. */
static GLuint
buffer_texture(struct gbm_buffer *buffer)
{
	bool fresh = buffer->texture == 0;

	if (fresh) {
		if (!buffer->cpu && buffer->image == EGL_NO_IMAGE_KHR)
			return 0;

		glGenTextures(1, &buffer->texture);
		glBindTexture(GL_TEXTURE_2D, buffer->texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		if (!buffer->cpu) {
			buffer->context->image_target_texture_2d(GL_TEXTURE_2D,
			                                         buffer->image);
			return buffer->texture;
		}
	}

	/* CPU-backed contents live in system memory and must be uploaded. */
	if (buffer->cpu && (fresh || buffer->dirty)) {
		uint32_t width = buffer->base.base.width;
		uint32_t height = buffer->base.base.height;
		uint32_t pitch = buffer->base.base.pitch;
		uint32_t row_pixels = pitch / 4;

		if (!buffer->base.base.map)
			return 0;

		glBindTexture(GL_TEXTURE_2D, buffer->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		/* Allocate storage once; refreshes are sub-image updates. */
		if (!buffer->tex_allocated) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA_EXT, width, height, 0,
			             GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
			buffer->tex_allocated = true;
		}

		if (row_pixels == width) {
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
			                GL_BGRA_EXT, GL_UNSIGNED_BYTE,
			                buffer->base.base.map);
		} else if (buffer->context->has_unpack_subimage) {
			glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT, row_pixels);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
			                GL_BGRA_EXT, GL_UNSIGNED_BYTE,
			                buffer->base.base.map);
			glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT, 0);
		} else {
			/* GLES2 without GL_EXT_unpack_subimage cannot skip padding. */
			uint32_t y;

			for (y = 0; y < height; ++y) {
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, width, 1, GL_BGRA_EXT,
				                GL_UNSIGNED_BYTE,
				                (uint8_t *)buffer->base.base.map
				                    + (size_t)y * pitch);
			}
		}

		buffer->dirty = false;
	}

	return buffer->texture;
}

/*
 * Map pixel coordinates onto clip space.
 *
 * No y flip: rendering into an FBO puts NDC -1 at the first row in memory,
 * which is also where wld puts the top of the image, so the two already agree.
 * The same reasoning makes glScissor's y match wld's y directly.
 */
static void
set_projection(struct gles_renderer *renderer, uint32_t width, uint32_t height)
{
	memset(renderer->proj, 0, sizeof renderer->proj);
	renderer->proj[0] = 2.0f / (GLfloat)width;
	renderer->proj[5] = 2.0f / (GLfloat)height;
	renderer->proj[10] = 1.0f;
	renderer->proj[12] = -1.0f;
	renderer->proj[13] = -1.0f;
	renderer->proj[15] = 1.0f;
}

static void
draw_quad(GLint pos_attrib, int32_t x1, int32_t y1, int32_t x2, int32_t y2)
{
	GLfloat verts[8] = {
		x1, y1,
		x2, y1,
		x1, y2,
		x2, y2,
	};

	glVertexAttribPointer(pos_attrib, 2, GL_FLOAT, GL_FALSE, 0, verts);
	glEnableVertexAttribArray(pos_attrib);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glDisableVertexAttribArray(pos_attrib);
}

static void
draw_textured_quad(struct gles_renderer *renderer, int32_t x1, int32_t y1,
                   int32_t x2, int32_t y2, GLfloat s1, GLfloat t1, GLfloat s2,
                   GLfloat t2, GLint pos_attrib, GLint tex_attrib)
{
	GLfloat verts[8] = {
		x1, y1,
		x2, y1,
		x1, y2,
		x2, y2,
	};
	GLfloat texcoords[8] = {
		s1, t1,
		s2, t1,
		s1, t2,
		s2, t2,
	};

	glVertexAttribPointer(pos_attrib, 2, GL_FLOAT, GL_FALSE, 0, verts);
	glEnableVertexAttribArray(pos_attrib);
	glVertexAttribPointer(tex_attrib, 2, GL_FLOAT, GL_FALSE, 0, texcoords);
	glEnableVertexAttribArray(tex_attrib);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glDisableVertexAttribArray(pos_attrib);
	glDisableVertexAttribArray(tex_attrib);
}

/*
 * Shared by copy_region and blend_region, which differ only in whether the
 * source is blended over the target or overwrites it.
 */
static void
composite_region(struct wld_renderer *base, struct buffer *src_base,
                 int32_t dst_x, int32_t dst_y, pixman_region32_t *region,
                 bool blend)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct gbm_buffer *src = gbm_buffer(&src_base->base);
	GLfloat src_w, src_h;
	GLfloat mul[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat add[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	pixman_box32_t *boxes;
	GLuint texture;
	int count;

	if (!renderer->target_texture)
		return;
	if (!(texture = buffer_texture(src)))
		return;

	src_w = (GLfloat)src_base->base.width;
	src_h = (GLfloat)src_base->base.height;

	/* An XRGB source carries no meaningful alpha, so force it opaque. */
	if (src_base->base.format == WLD_FORMAT_XRGB8888) {
		mul[3] = 0.0f;
		add[3] = 1.0f;
	}

	glUseProgram(renderer->textured.program);
	glUniformMatrix4fv(renderer->textured.proj, 1, GL_FALSE, renderer->proj);
	glUniform4fv(renderer->textured.mul, 1, mul);
	glUniform4fv(renderer->textured.add, 1, add);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(renderer->textured.tex, 0);

	if (blend) {
		glEnable(GL_BLEND);
		/* wld buffers hold premultiplied alpha. */
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glDisable(GL_BLEND);
	}

	boxes = pixman_region32_rectangles(region, &count);
	while (count--) {
		draw_textured_quad(renderer, boxes->x1 + dst_x, boxes->y1 + dst_y,
		                   boxes->x2 + dst_x, boxes->y2 + dst_y,
		                   boxes->x1 / src_w, boxes->y1 / src_h,
		                   boxes->x2 / src_w, boxes->y2 / src_h,
		                   renderer->textured.pos, renderer->textured.texcoord);
		++boxes;
	}

	glDisable(GL_BLEND);
}

uint32_t
renderer_capabilities(struct wld_renderer *base, struct buffer *buffer)
{
	/*
	 * Anything we can hang an EGLImage off is both readable and writable by
	 * the GPU. This is what lets the compositor draw client dmabufs, which
	 * the pixman renderer has to skip.
	 */
	if (buffer->base.impl == &wld_buffer_impl)
		return WLD_CAPABILITY_READ | WLD_CAPABILITY_WRITE;

	return 0;
}

bool
renderer_set_target(struct wld_renderer *base, struct buffer *buffer)
{
	struct gles_renderer *renderer = gles_renderer(base);
	GLuint texture;

	if (!buffer) {
		renderer->target_texture = 0;
		return true;
	}

	if (buffer->base.impl != &wld_buffer_impl)
		return false;

	if (!(texture = buffer_texture(gbm_buffer(&buffer->base))))
		return false;

	glBindFramebuffer(GL_FRAMEBUFFER, renderer->fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
	                       texture, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		DEBUG("framebuffer incomplete for target buffer\n");
		renderer->target_texture = 0;
		return false;
	}

	renderer->target_texture = texture;
	renderer->target_width = buffer->base.width;
	renderer->target_height = buffer->base.height;
	glViewport(0, 0, buffer->base.width, buffer->base.height);
	set_projection(renderer, buffer->base.width, buffer->base.height);

	return true;
}

void
renderer_fill_rectangle(struct wld_renderer *base, uint32_t color, int32_t x,
                        int32_t y, uint32_t width, uint32_t height)
{
	struct gles_renderer *renderer = gles_renderer(base);
	GLfloat rgba[4];

	if (!renderer->target_texture)
		return;

	color_to_gl(color, rgba);
	glDisable(GL_BLEND);
	glUseProgram(renderer->solid.program);
	glUniformMatrix4fv(renderer->solid.proj, 1, GL_FALSE, renderer->proj);
	glUniform4fv(renderer->solid.color, 1, rgba);
	draw_quad(renderer->solid.pos, x, y, x + (int32_t)width,
	          y + (int32_t)height);
}

void
renderer_fill_region(struct wld_renderer *base, uint32_t color,
                     pixman_region32_t *region)
{
	struct gles_renderer *renderer = gles_renderer(base);
	pixman_box32_t *boxes;
	GLfloat rgba[4];
	int count;

	if (!renderer->target_texture)
		return;

	color_to_gl(color, rgba);
	glDisable(GL_BLEND);
	glUseProgram(renderer->solid.program);
	glUniformMatrix4fv(renderer->solid.proj, 1, GL_FALSE, renderer->proj);
	glUniform4fv(renderer->solid.color, 1, rgba);

	boxes = pixman_region32_rectangles(region, &count);
	while (count--) {
		draw_quad(renderer->solid.pos, boxes->x1, boxes->y1, boxes->x2,
		          boxes->y2);
		++boxes;
	}
}

void
renderer_copy_rectangle(struct wld_renderer *base, struct buffer *src,
                        int32_t dst_x, int32_t dst_y, int32_t src_x,
                        int32_t src_y, uint32_t width, uint32_t height)
{
	pixman_region32_t region;

	pixman_region32_init_rect(&region, src_x, src_y, width, height);
	composite_region(base, src, dst_x - src_x, dst_y - src_y, &region, false);
	pixman_region32_fini(&region);
}

void
renderer_copy_region(struct wld_renderer *base, struct buffer *src,
                     int32_t dst_x, int32_t dst_y, pixman_region32_t *region)
{
	composite_region(base, src, dst_x, dst_y, region, false);
}

void
renderer_blend_region(struct wld_renderer *base, struct buffer *src,
                      int32_t dst_x, int32_t dst_y, pixman_region32_t *region)
{
	composite_region(base, src, dst_x, dst_y, region, true);
}

/*
 * Glyphs are cached as individual GL_ALPHA textures in a small direct-mapped
 * table. Freetype hands us the same stable struct glyph * for a given glyph,
 * so the pointer is the key.
 */
static struct glyph_entry *
glyph_texture(struct gles_renderer *renderer, struct glyph *glyph)
{
	FT_Bitmap *bitmap = &glyph->bitmap;
	struct glyph_entry *entry;
	uint8_t *pixels;
	uint32_t row, col;
	size_t index;

	index = ((uintptr_t)glyph >> 4) % GLYPH_CACHE_SIZE;
	entry = &renderer->glyphs[index];

	if (entry->glyph == glyph)
		return entry->texture ? entry : NULL;

	if (entry->texture)
		glDeleteTextures(1, &entry->texture);
	entry->glyph = glyph;
	entry->texture = 0;
	entry->width = bitmap->width;
	entry->height = bitmap->rows;

	if (bitmap->width == 0 || bitmap->rows == 0)
		return NULL;

	if (!(pixels = malloc((size_t)bitmap->width * bitmap->rows)))
		return NULL;

	switch (bitmap->pixel_mode) {
	case FT_PIXEL_MODE_GRAY:
		for (row = 0; row < bitmap->rows; ++row) {
			memcpy(pixels + (size_t)row * bitmap->width,
			       bitmap->buffer + (size_t)row * bitmap->pitch,
			       bitmap->width);
		}
		break;
	case FT_PIXEL_MODE_MONO:
		for (row = 0; row < bitmap->rows; ++row) {
			const uint8_t *src = bitmap->buffer + (size_t)row * bitmap->pitch;
			uint8_t *dst = pixels + (size_t)row * bitmap->width;

			for (col = 0; col < bitmap->width; ++col)
				dst[col] = (src[col >> 3] & (0x80 >> (col & 7))) ? 0xff : 0x00;
		}
		break;
	default:
		free(pixels);
		return NULL;
	}

	glGenTextures(1, &entry->texture);
	glBindTexture(GL_TEXTURE_2D, entry->texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, bitmap->width, bitmap->rows, 0,
	             GL_ALPHA, GL_UNSIGNED_BYTE, pixels);

	free(pixels);

	return entry;
}

void
renderer_draw_text(struct wld_renderer *base, struct font *font, uint32_t color,
                   int32_t x, int32_t y, const char *text, uint32_t length,
                   struct wld_extents *extents)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct glyph_entry *entry;
	struct glyph *glyph;
	FT_UInt glyph_index;
	GLfloat rgba[4];
	uint32_t origin_x = 0, c;
	int ret;

	if (!renderer->target_texture)
		return;

	if (length == -1)
		length = strlen(text);

	color_to_gl(color, rgba);

	glUseProgram(renderer->glyph.program);
	glUniformMatrix4fv(renderer->glyph.proj, 1, GL_FALSE, renderer->proj);
	glUniform4fv(renderer->glyph.color, 1, rgba);
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(renderer->glyph.tex, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	while ((ret = FcUtf8ToUcs4((FcChar8 *)text, &c, length)) > 0 && c != '\0') {
		text += ret;
		length -= ret;
		glyph_index = FT_Get_Char_Index(font->face, c);

		if (!font_ensure_glyph(font, glyph_index))
			continue;

		glyph = font->glyphs[glyph_index];

		if ((entry = glyph_texture(renderer, glyph))) {
			int32_t gx = x + (int32_t)origin_x + glyph->x;
			int32_t gy = y + glyph->y;

			glBindTexture(GL_TEXTURE_2D, entry->texture);
			draw_textured_quad(renderer, gx, gy, gx + (int32_t)entry->width,
			                   gy + (int32_t)entry->height, 0.0f, 0.0f, 1.0f,
			                   1.0f, renderer->glyph.pos,
			                   renderer->glyph.texcoord);
		}

		origin_x += glyph->advance;
	}

	glDisable(GL_BLEND);

	if (extents)
		extents->advance = origin_x;
}

bool
renderer_read_pixels(struct wld_renderer *base, int32_t x, int32_t y,
                     uint32_t width, uint32_t height, uint32_t pitch,
                     void *data)
{
	struct gles_renderer *renderer = gles_renderer(base);
	uint32_t row_bytes = width * 4;

	if (!renderer->target_texture)
		return false;

	glFinish();

	/*
	 * GLES2 has no GL_PACK_ROW_LENGTH, so a destination with padding has to
	 * be filled a row at a time. glReadPixels' y origin matches ours, since
	 * rendering into an FBO puts window y 0 at the first row in memory.
	 */
	if (pitch == row_bytes) {
		glReadPixels(x, y, width, height, GL_BGRA_EXT, GL_UNSIGNED_BYTE, data);
	} else {
		uint32_t row;

		for (row = 0; row < height; ++row) {
			glReadPixels(x, y + (int32_t)row, width, 1, GL_BGRA_EXT,
			             GL_UNSIGNED_BYTE, (uint8_t *)data + (size_t)row * pitch);
		}
	}

	return glGetError() == GL_NO_ERROR;
}

void
renderer_flush(struct wld_renderer *base)
{
	struct gles_renderer *renderer = gles_renderer(base);

	(void)renderer;
	/*
	 * Callers use flush() as a barrier before handing a buffer to KMS or
	 * reading it on the CPU, so this has to be a real finish, not glFlush.
	 */
	glFinish();
}

void
renderer_destroy(struct wld_renderer *base)
{
	struct gles_renderer *renderer = gles_renderer(base);
	size_t i;

	for (i = 0; i < GLYPH_CACHE_SIZE; ++i) {
		if (renderer->glyphs[i].texture)
			glDeleteTextures(1, &renderer->glyphs[i].texture);
	}

	glDeleteProgram(renderer->solid.program);
	glDeleteProgram(renderer->textured.program);
	glDeleteProgram(renderer->glyph.program);
	glDeleteFramebuffers(1, &renderer->fbo);
	free(renderer);
}

struct wld_renderer *
context_create_renderer(struct wld_context *base)
{
	struct gbm_context *context = gbm_context(base);
	struct gles_renderer *renderer;

	if (!(renderer = calloc(1, sizeof *renderer)))
		return NULL;

	renderer->context = context;

	renderer->solid.program = link_program(vertex_solid_src, fragment_solid_src);
	renderer->textured.program = link_program(vertex_tex_src, fragment_tex_src);
	renderer->glyph.program = link_program(vertex_tex_src, fragment_glyph_src);

	if (!renderer->solid.program || !renderer->textured.program
	    || !renderer->glyph.program) {
		goto error;
	}

	renderer->solid.pos = glGetAttribLocation(renderer->solid.program, "pos");
	renderer->solid.proj = glGetUniformLocation(renderer->solid.program, "proj");
	renderer->solid.color =
	    glGetUniformLocation(renderer->solid.program, "color");

	renderer->textured.pos =
	    glGetAttribLocation(renderer->textured.program, "pos");
	renderer->textured.texcoord =
	    glGetAttribLocation(renderer->textured.program, "texcoord");
	renderer->textured.proj =
	    glGetUniformLocation(renderer->textured.program, "proj");
	renderer->textured.tex =
	    glGetUniformLocation(renderer->textured.program, "tex");
	renderer->textured.mul =
	    glGetUniformLocation(renderer->textured.program, "mul");
	renderer->textured.add =
	    glGetUniformLocation(renderer->textured.program, "add");

	renderer->glyph.pos = glGetAttribLocation(renderer->glyph.program, "pos");
	renderer->glyph.texcoord =
	    glGetAttribLocation(renderer->glyph.program, "texcoord");
	renderer->glyph.proj =
	    glGetUniformLocation(renderer->glyph.program, "proj");
	renderer->glyph.tex = glGetUniformLocation(renderer->glyph.program, "tex");
	renderer->glyph.color =
	    glGetUniformLocation(renderer->glyph.program, "color");

	glGenFramebuffers(1, &renderer->fbo);

	renderer_initialize(&renderer->base, &wld_renderer_impl);

	return &renderer->base;

error:
	if (renderer->solid.program)
		glDeleteProgram(renderer->solid.program);
	if (renderer->textured.program)
		glDeleteProgram(renderer->textured.program);
	if (renderer->glyph.program)
		glDeleteProgram(renderer->glyph.program);
	free(renderer);
	return NULL;
}
