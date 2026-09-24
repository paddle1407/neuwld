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

#define _DEFAULT_SOURCE 1

#include "drm-private.h"
#include "drm.h"
#include "wld-private.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <poll.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <drm_fourcc.h>
#include <gbm.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include <fontconfig/fontconfig.h>

/* Past this many boxes, a dirty region is uploaded as its bounding box. */
#define DIRTY_BOX_LIMIT 16

/*
 * Glyphs live in one alpha atlas per context, shared by every renderer on it.
 * The table is open-addressed on the glyph's serial; once it is three quarters
 * full or the atlas has no room left, both are emptied and refilled on demand,
 * which bounds the cache at one texture and one table whatever is drawn.
 */
#define GLYPH_ATLAS_SIZE 1024
#define GLYPH_TABLE_SIZE 4096
#define GLYPH_TABLE_LOAD (GLYPH_TABLE_SIZE / 4 * 3)
/* Blank texels between glyphs, so filtering never reaches a neighbour. */
#define GLYPH_PADDING 1

/* Fixed attribute locations, bound before linking, shared by all programs. */
#define ATTRIB_POS 0
#define ATTRIB_TEXCOORD 1
#define ATTRIB_COLOR 2
/* Every vertex is x, y, s, t, r, g, b, a; a quad is two triangles. */
#define VERTEX_FLOATS 8
#define QUAD_FLOATS (6 * VERTEX_FLOATS)

/* Keep window-sized CPU pixel storage out of malloc's retained heap arenas.
 * Small images still use calloc to avoid a mapping/page per tiny asset. */
#define PIXEL_MAPPING_THRESHOLD (256u * 1024u)

struct glyph_slot {
	uint64_t serial; /* 0 marks an empty slot; serials start at 1 */
	uint16_t x, y, width, height;
};

struct glyph_atlas {
	GLuint texture;
	uint32_t size;
	/* Shelf packing: glyphs fill a row left to right, rows stack down. */
	uint32_t shelf_x, shelf_y, shelf_height;
	unsigned count;
	struct glyph_slot *slots;
};

struct gles_renderer;

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
	PFNEGLCREATESYNCKHRPROC create_sync;
	PFNEGLDESTROYSYNCKHRPROC destroy_sync;
	PFNEGLWAITSYNCKHRPROC wait_sync;
	PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_native_fence_fd;

	bool has_modifiers;
	bool has_unpack_subimage;
	/* Set when a client's DRM sync_file fence can be waited on by the GPU. */
	bool has_fence_sync;
	/* Set when our own rendering can be exported as a sync_file. */
	bool has_native_fence;
	/*
	 * Set while rendering has been submitted without being waited for, which
	 * a CPU mapping of one of our buffers has to catch up on first.
	 */
	bool unfinished;

	/*
	 * GL state belongs to the context, and every renderer created from it
	 * shares that one context, so what is currently bound is tracked here.
	 * Changes go through the helpers below, which skip redundant calls.
	 */
	struct {
		GLuint program, texture, fbo;
		int blend; /* -1 until first set */
		int scissor; /* -1 until first set */
		pixman_box32_t scissor_box;
	} gl;

	/*
	 * Quads are staged here and drawn in one call per batch. A batch is
	 * everything staged since the last draw, under the GL state it was staged
	 * for; whatever would change that state draws the batch first, so calls
	 * that share a program, texture and blend mode -- a titlebar's hundred
	 * button pixels, a border's four sides -- cost one draw between them.
	 */
	GLuint vbo;
	GLfloat *verts;
	size_t verts_capacity; /* in quads */
	struct {
		struct gles_renderer *renderer;
		GLuint program, texture;
		bool blend;
		size_t quads;
	} batch;

	/*
	 * Counted up whenever work is queued on the GPU. A fence exported since
	 * the last count still covers everything queued, so it is handed out
	 * again rather than created anew for every buffer a client replaces.
	 */
	uint64_t submissions, fence_submissions;
	int fence_fd;

	/* Every renderer on this context, so that a texture being deleted can be
	 * forgotten as a render target wherever it was one. */
	struct gles_renderer *renderers;

	struct glyph_atlas atlas;
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
	 * An imported buffer has no gbm_bo to ask for its layout, so keep the
	 * modifier the client declared. Reporting LINEAR for a tiled import
	 * would build a DRM framebuffer that scans out garbage. Buffers we
	 * allocate without a bo are linear dumb buffers, and calloc already
	 * leaves this at DRM_FORMAT_MOD_LINEAR (0) for them.
	 */
	uint64_t modifier;

	/*
	 * WLD_FLAG_MAP buffers use cached CPU memory (DRM dumb for cursors).
	 * NVIDIA will not sample or render a linear dmabuf, so these are uploaded
	 * into an ordinary GL texture instead of being wrapped in an EGLImage.
	 */
	bool cpu;
	bool dumb;
	bool tex_allocated;
	/* Allocated for KMS, which is fenced by wld_export_fence(), not flush. */
	bool scanout;
	/* Pixels only ever arrive through wld_buffer_upload(): there is no CPU
	 * copy of them here, so nothing to map and nothing to keep in step. */
	bool upload_only;
	size_t mapping_size;

	/*
	 * What of a CPU buffer's texture is stale. `dirty` covers all of it;
	 * otherwise only `dirty_region` has to be uploaded again. A CPU renderer
	 * reports what it drew through buffer_damage() before it flushes, and
	 * `damage_reported` tells that flush it need not assume the worst.
	 */
	bool dirty;
	bool damage_reported;
	pixman_region32_t dirty_region;
};

struct gles_renderer {
	struct wld_renderer base;
	struct gbm_context *context;

	GLuint fbo;
	GLuint target_texture;
	uint32_t target_width, target_height;

	/*
	 * Uniforms are program state, and the programs are this renderer's own,
	 * so the values last loaded into them are tracked here. Colour is not a
	 * uniform but a vertex attribute, so that fills of different colours can
	 * share one draw.
	 */
	struct {
		GLuint program;
		GLint proj;
		bool proj_valid;
	} solid;
	struct {
		GLuint program;
		GLint proj, mul, add;
		bool proj_valid;
		int opaque; /* -1 until mul/add are first loaded */
	} textured;
	struct {
		GLuint program;
		GLint proj;
		bool proj_valid;
	} glyph;

	GLfloat proj[16];

	/* wld_set_clip(): a box outside of which nothing is drawn, until the
	 * target changes. Applied as the scissor when this renderer draws. */
	bool clip_enabled;
	pixman_box32_t clip;

	struct gles_renderer *next;
};

#define CONTEXT_IMPLEMENTS_QUERY_MODIFIERS
#define BUFFER_IMPLEMENTS_FLUSH
#define BUFFER_IMPLEMENTS_DAMAGE
#define BUFFER_IMPLEMENTS_UPLOAD
#define RENDERER_IMPLEMENTS_REGION
#define RENDERER_IMPLEMENTS_SET_CLIP
#define RENDERER_IMPLEMENTS_BLEND
#define RENDERER_IMPLEMENTS_BLEND_SCALED
#define RENDERER_IMPLEMENTS_READ_PIXELS
#define RENDERER_IMPLEMENTS_WAIT_FENCE
#define RENDERER_IMPLEMENTS_EXPORT_FENCE
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
    "attribute vec4 color;\n"
    "varying vec4 v_color;\n"
    "void main() {\n"
    "  v_color = color;\n"
    "  gl_Position = proj * vec4(pos, 0.0, 1.0);\n"
    "}\n";

static const char fragment_solid_src[] =
    "precision mediump float;\n"
    "varying vec4 v_color;\n"
    "void main() { gl_FragColor = v_color; }\n";

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

static const char vertex_glyph_src[] =
    "uniform mat4 proj;\n"
    "attribute vec2 pos;\n"
    "attribute vec2 texcoord;\n"
    "attribute vec4 color;\n"
    "varying vec2 v_tex;\n"
    "varying vec4 v_color;\n"
    "void main() {\n"
    "  v_tex = texcoord;\n"
    "  v_color = color;\n"
    "  gl_Position = proj * vec4(pos, 0.0, 1.0);\n"
    "}\n";

static const char fragment_glyph_src[] =
    "precision mediump float;\n"
    "varying vec2 v_tex;\n"
    "varying vec4 v_color;\n"
    "uniform sampler2D tex;\n"
    "void main() { gl_FragColor = v_color * texture2D(tex, v_tex).a; }\n";

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
	/* The vertex layout is shared, so every program reads it from the same
	 * slots; a name a shader does not use is simply ignored. */
	glBindAttribLocation(program, ATTRIB_POS, "pos");
	glBindAttribLocation(program, ATTRIB_TEXCOORD, "texcoord");
	glBindAttribLocation(program, ATTRIB_COLOR, "color");
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
	 * Claim everything and let driver_create_context() fail if the stack is
	 * not actually usable; drm.c then falls through to the next driver.
	 *
	 * WLD_DRM_NO_GBM still disables this backend - drm.c applies that to
	 * every driver uniformly, so it is not handled here.
	 */
	return true;
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

static EGLDisplay
gbm_egl_display(struct gbm_device *device)
{
	const char *extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	bool khr = has_extension(extensions, "EGL_KHR_platform_gbm");
	bool mesa = has_extension(extensions, "EGL_MESA_platform_gbm");

	/* A gbm_device is not an X display or a wl_display. The legacy
	 * eglGetDisplay API leaves native-platform detection to the EGL loader
	 * (and environment); request GBM explicitly for multi-vendor systems. */
	if (khr) {
		PFNEGLGETPLATFORMDISPLAYPROC get_display =
		    (PFNEGLGETPLATFORMDISPLAYPROC)eglGetProcAddress("eglGetPlatformDisplay");
		if (get_display)
			return get_display(EGL_PLATFORM_GBM_KHR, device, NULL);
	}
	if ((khr || mesa) && has_extension(extensions, "EGL_EXT_platform_base")) {
		PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
		    (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
		if (get_display)
			return get_display(EGL_PLATFORM_GBM_KHR, device, NULL);
	}
	fprintf(stderr, "wld: GBM initialization failed: EGL has no GBM platform entry point\n");
	return EGL_NO_DISPLAY;
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

	if (!(context = calloc(1, sizeof *context)))
		return NULL;

	context->fd = drm_fd;

	if (!(context->gbm = gbm_create_device(drm_fd))) {
		fprintf(stderr, "wld: gbm_create_device failed: %s\n", strerror(errno));
		goto error0;
	}

	context->display = gbm_egl_display(context->gbm);
	if (context->display == EGL_NO_DISPLAY) {
		fprintf(stderr, "wld: GBM platform display failed (EGL error 0x%x)\n", eglGetError());
		goto error1;
	}

	if (!eglInitialize(context->display, &major, &minor)) {
		fprintf(stderr, "wld: GBM eglInitialize failed (EGL error 0x%x)\n", eglGetError());
		goto error1;
	}

	extensions = eglQueryString(context->display, EGL_EXTENSIONS);
	if (!has_extension(extensions, "EGL_KHR_image_base")
	    || !has_extension(extensions, "EGL_EXT_image_dma_buf_import")) {
		fprintf(stderr, "wld: GBM initialization failed: required EGL image extensions missing\n");
		goto error2;
	}
	context->has_modifiers =
	    has_extension(extensions, "EGL_EXT_image_dma_buf_import_modifiers");

	if (!eglBindAPI(EGL_OPENGL_ES_API)) {
		fprintf(stderr, "wld: GBM eglBindAPI failed (EGL error 0x%x)\n", eglGetError());
		goto error2;
	}

	/*
	 * We only ever render to FBOs backed by EGLImages, never to an EGL
	 * surface, so a config is unnecessary and a surfaceless context is
	 * enough.
	 */
	if (!has_extension(extensions, "EGL_KHR_surfaceless_context")) {
		fprintf(stderr, "wld: GBM initialization failed: EGL_KHR_surfaceless_context missing\n");
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
			fprintf(stderr, "wld: GBM eglChooseConfig found no usable config (EGL error 0x%x)\n", eglGetError());
			goto error2;
		}
	}

	context->context = eglCreateContext(context->display, config,
	                                    EGL_NO_CONTEXT, context_attribs);
	if (context->context == EGL_NO_CONTEXT) {
		fprintf(stderr, "wld: GBM eglCreateContext failed (EGL error 0x%x)\n", eglGetError());
		goto error2;
	}

	if (!eglMakeCurrent(context->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
	                    context->context)) {
		fprintf(stderr, "wld: GBM eglMakeCurrent failed (EGL error 0x%x)\n", eglGetError());
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

	/*
	 * Explicit synchronization: turn a client's sync_file into an EGLSync the
	 * GPU can wait on, so composition is ordered after the client's rendering
	 * on drivers that do not provide implicit fences for dmabufs.
	 */
	if (has_extension(extensions, "EGL_KHR_fence_sync")
	    && has_extension(extensions, "EGL_ANDROID_native_fence_sync")
	    && has_extension(extensions, "EGL_KHR_wait_sync")) {
		context->create_sync =
		    (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
		context->destroy_sync =
		    (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
		context->wait_sync =
		    (PFNEGLWAITSYNCKHRPROC)eglGetProcAddress("eglWaitSyncKHR");
		context->has_fence_sync = context->create_sync
		    && context->destroy_sync && context->wait_sync;
		context->dup_native_fence_fd = (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)
		    eglGetProcAddress("eglDupNativeFenceFDANDROID");
		context->has_native_fence =
		    context->has_fence_sync && context->dup_native_fence_fd;
	}
	if (!context->has_fence_sync) {
		fprintf(stderr, "wld: GBM has no EGL fence sync; explicit client "
		                "synchronization is unavailable\n");
	}

	if (!context->create_image || !context->destroy_image
	    || !context->image_target_texture_2d) {
		fprintf(stderr, "wld: GBM initialization failed: required EGL/GL entry points missing\n");
		goto error3;
	}

	{
		const char *gl_ext = (const char *)glGetString(GL_EXTENSIONS);
		context->has_unpack_subimage =
		    has_extension(gl_ext, "GL_EXT_unpack_subimage");
	}

	/*
	 * State nothing ever changes is set once: one vertex buffer with a fixed
	 * layout, one texture unit, and premultiplied-alpha blending whenever
	 * blending is on at all.
	 */
	glGenBuffers(1, &context->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, context->vbo);
	glVertexAttribPointer(ATTRIB_POS, 2, GL_FLOAT, GL_FALSE,
	                      VERTEX_FLOATS * sizeof(GLfloat), (void *)0);
	glVertexAttribPointer(ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE,
	                      VERTEX_FLOATS * sizeof(GLfloat),
	                      (void *)(2 * sizeof(GLfloat)));
	glVertexAttribPointer(ATTRIB_COLOR, 4, GL_FLOAT, GL_FALSE,
	                      VERTEX_FLOATS * sizeof(GLfloat),
	                      (void *)(4 * sizeof(GLfloat)));
	glEnableVertexAttribArray(ATTRIB_POS);
	glEnableVertexAttribArray(ATTRIB_TEXCOORD);
	glEnableVertexAttribArray(ATTRIB_COLOR);
	glActiveTexture(GL_TEXTURE0);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_BLEND);
	context->gl.blend = 0;
	context->gl.scissor = -1;
	context->fence_fd = -1;

	context_initialize(&context->base, &wld_context_impl);
	DEBUG("using GBM/EGL context (EGL %d.%d)\n", major, minor);
	fprintf(stderr, "wld: EGL %d.%d, GL renderer: %s, vendor: %s\n",
	        major, minor, (const char *)glGetString(GL_RENDERER),
	        (const char *)glGetString(GL_VENDOR));

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

/**** GL state ****/

static void
use_program(struct gbm_context *context, GLuint program)
{
	if (context->gl.program != program) {
		glUseProgram(program);
		context->gl.program = program;
	}
}

static void
bind_texture(struct gbm_context *context, GLuint texture)
{
	if (context->gl.texture != texture) {
		glBindTexture(GL_TEXTURE_2D, texture);
		context->gl.texture = texture;
	}
}

static void
set_blend(struct gbm_context *context, bool blend)
{
	if (context->gl.blend != blend) {
		if (blend)
			glEnable(GL_BLEND);
		else
			glDisable(GL_BLEND);
		context->gl.blend = blend;
	}
}

/**** Batching ****/

/*
 * Room for `quads` quads in the staging array. Returns NULL, having drawn
 * nothing, if that much memory cannot be had.
 */
static GLfloat *
reserve_quads(struct gbm_context *context, size_t quads)
{
	GLfloat *verts;
	size_t capacity;

	if (quads <= context->verts_capacity)
		return context->verts;

	capacity = context->verts_capacity ? context->verts_capacity : 64;
	while (capacity < quads) {
		if (capacity > SIZE_MAX / 2 / (QUAD_FLOATS * sizeof(GLfloat)))
			return NULL;
		capacity *= 2;
	}

	if (!(verts = realloc(context->verts,
	                      capacity * QUAD_FLOATS * sizeof(GLfloat))))
		return NULL;

	context->verts = verts;
	context->verts_capacity = capacity;
	return verts;
}

static GLfloat *
put_quad(GLfloat *v, GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2,
         GLfloat s1, GLfloat t1, GLfloat s2, GLfloat t2, const GLfloat rgba[4])
{
	const GLfloat r = rgba[0], g = rgba[1], b = rgba[2], a = rgba[3];
	const GLfloat quad[QUAD_FLOATS] = {
		x1, y1, s1, t1, r, g, b, a,
		x2, y1, s2, t1, r, g, b, a,
		x1, y2, s1, t2, r, g, b, a,
		x2, y1, s2, t1, r, g, b, a,
		x2, y2, s2, t2, r, g, b, a,
		x1, y2, s1, t2, r, g, b, a,
	};

	memcpy(v, quad, sizeof quad);
	return v + QUAD_FLOATS;
}

/* Draw whatever is staged, with the state it was staged under. */
static void
batch_flush(struct gbm_context *context)
{
	if (!context->batch.quads)
		return;

	glBufferData(GL_ARRAY_BUFFER,
	             context->batch.quads * QUAD_FLOATS * sizeof(GLfloat),
	             context->verts, GL_STREAM_DRAW);
	glDrawArrays(GL_TRIANGLES, 0, context->batch.quads * 6);
	context->batch.quads = 0;
	++context->submissions;
}

/*
 * Put the state the next quads need in place. Quads staged under anything
 * else are drawn first; under the same state they simply wait for company.
 */
static void
batch_begin(struct gles_renderer *renderer, GLuint program, GLuint texture,
            bool blend)
{
	struct gbm_context *context = renderer->context;

	if (context->batch.quads
	    && (context->batch.renderer != renderer
	        || context->batch.program != program
	        || context->batch.texture != texture
	        || context->batch.blend != blend))
		batch_flush(context);

	use_program(context, program);
	bind_texture(context, texture);
	set_blend(context, blend);
	context->batch.renderer = renderer;
	context->batch.program = program;
	context->batch.texture = texture;
	context->batch.blend = blend;
}

/* Where to write `quads` more quads for the current batch, or NULL. */
static GLfloat *
batch_reserve(struct gbm_context *context, size_t quads)
{
	GLfloat *v = reserve_quads(context, context->batch.quads + quads);

	if (!v)
		return NULL;
	v += context->batch.quads * QUAD_FLOATS;
	context->batch.quads += quads;
	return v;
}

/*
 * Bind a texture in order to change it. Quads staged against it, or against
 * anything else, are drawn first: the binding is part of their state, and
 * GL only orders a texture update after draws already issued.
 */
static void
bind_texture_for_upload(struct gbm_context *context, GLuint texture)
{
	batch_flush(context);
	bind_texture(context, texture);
}

/* Deleting a bound texture unbinds it, and its name can be handed out again. */
static void
delete_texture(struct gbm_context *context, GLuint texture)
{
	struct gles_renderer *renderer;

	if (context->batch.quads && context->batch.texture == texture)
		batch_flush(context);
	glDeleteTextures(1, &texture);
	if (context->gl.texture == texture)
		context->gl.texture = 0;
	/* The name may come back for another buffer, which must not then pass
	 * for the target that is already attached. */
	for (renderer = context->renderers; renderer; renderer = renderer->next) {
		if (renderer->target_texture == texture)
			renderer->target_texture = 0;
	}
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
		if (buffer->bo) {
			object->u64 = gbm_bo_get_modifier(buffer->bo);
			return true;
		}
		/* An import keeps the layout it was given; anything else we
		 * allocated ourselves without a bo is a linear dumb buffer. */
		object->u64 = buffer->modifier != DRM_FORMAT_MOD_INVALID
		                  ? buffer->modifier
		                  : DRM_FORMAT_MOD_LINEAR;
		return true;
	case WLD_DRM_OBJECT_PRIME_FD:
		/* Whoever gets the descriptor has no fence to go by. */
		if (buffer->context->unfinished) {
			glFinish();
			buffer->context->unfinished = false;
		}
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
	pixman_region32_init(&buffer->dirty_region);
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
	if (flags & (WLD_FLAG_MAP | WLD_FLAG_UPLOAD)) {
		struct gbm_buffer *cpu_buffer;
		uint32_t bytes_per_pixel = format_bytes_per_pixel(format);
		if (!width || !height || !bytes_per_pixel ||
		    width > UINT32_MAX / bytes_per_pixel)
			return NULL;
		uint32_t pitch = width * bytes_per_pixel;
		if (height > SIZE_MAX / pitch)
			return NULL;
		size_t size = (size_t)height * pitch;
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

		/*
		 * A texture and nothing else: the pixels come straight from the
		 * caller's memory through wld_buffer_upload(), so there is no copy
		 * of them to keep here, and none to make on the way in.
		 */
		if (flags & WLD_FLAG_UPLOAD) {
			buffer = new_buffer(context, NULL, EGL_NO_IMAGE_KHR, 0, false,
			                    width, height, format, pitch);
			if (!buffer)
				return NULL;
			cpu_buffer = gbm_buffer(&buffer->base);
			cpu_buffer->cpu = true;
			cpu_buffer->upload_only = true;
			return buffer;
		}

		bool mapped = size >= PIXEL_MAPPING_THRESHOLD;
		if (mapped) {
			/* Anonymous mappings are zero-filled, cached RAM just like calloc,
			 * but munmap returns their pages even when small heap objects live
			 * longer than the terminal's upload buffer. */
			data = mmap(NULL, size, PROT_READ | PROT_WRITE,
			            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
			if (data == MAP_FAILED) return NULL;
		} else if (!(data = calloc(height, pitch))) {
			return NULL;
		}

		buffer = new_buffer(context, NULL, EGL_NO_IMAGE_KHR, 0, false, width,
		                    height, format, pitch);
		if (!buffer) {
			if (mapped) munmap(data, size);
			else free(data);
			return NULL;
		}
		cpu_buffer = gbm_buffer(&buffer->base);
		cpu_buffer->mapping_size = mapped ? size : 0;

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

	gbm_buffer(&buffer->base)->scanout = flags & WLD_DRM_FLAG_SCANOUT;
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
		/* new_buffer took no ownership, so the handle is still ours. */
		if (handle) {
			struct drm_gem_close close_arg = { .handle = handle };
			drmIoctl(context->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
		}
		return NULL;
	}

	gbm_buffer(&buffer->base)->modifier = modifier;

	return buffer;
}

int
context_query_modifiers(struct wld_context *base, uint32_t format,
                        uint64_t *modifiers, int max)
{
	struct gbm_context *context = gbm_context(base);
	EGLint count = 0, capacity;
	EGLuint64KHR *available;
	EGLBoolean *external_only;
	int written = 0;

	if (!context->query_dmabuf_modifiers || !context->has_modifiers)
		return -1;
	if (max <= 0)
		return 0;
	if (!context->query_dmabuf_modifiers(context->display, format, 0, NULL, NULL,
	                                     &count))
		return -1;
	if (count <= 0)
		return 0;
	capacity = count;
	available = calloc(capacity, sizeof(*available));
	external_only = calloc(capacity, sizeof(*external_only));
	if (!available || !external_only) {
		written = -1;
		goto done;
	}
	if (!context->query_dmabuf_modifiers(context->display, format, capacity,
	                                     available, external_only, &count)) {
		written = -1;
		goto done;
	}
	/* Our shaders sample GL_TEXTURE_2D, not GL_TEXTURE_EXTERNAL_OES. */
	for (int i = 0; i < count && i < capacity && written < max; ++i) {
		if (!external_only[i])
			modifiers[written++] = available[i];
	}
done:
	free(external_only);
	free(available);
	return written;
}

void
context_destroy(struct wld_context *base)
{
	struct gbm_context *context = gbm_context(base);

	if (context->atlas.texture)
		glDeleteTextures(1, &context->atlas.texture);
	free(context->atlas.slots);
	glDeleteBuffers(1, &context->vbo);
	free(context->verts);
	if (context->fence_fd >= 0)
		close(context->fence_fd);

	eglMakeCurrent(context->display, EGL_NO_SURFACE, EGL_NO_SURFACE,
	               EGL_NO_CONTEXT);
	eglDestroyContext(context->display, context->context);
	eglTerminate(context->display);
	gbm_device_destroy(context->gbm);
	/* The fd belongs to the caller of wld_drm_create_context, which closes
	 * it itself; closing it here would close it twice. */
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

	/* The GPU may still be writing it; see renderer_flush(). */
	if (buffer->context->unfinished) {
		glFinish();
		buffer->context->unfinished = false;
	}

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
buffer_damage(struct buffer *base, pixman_region32_t *region)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	if (!buffer->cpu || buffer->upload_only)
		return;

	buffer->damage_reported = true;
	if (!region) {
		buffer->dirty = true;
		return;
	}

	pixman_region32_union(&buffer->dirty_region, &buffer->dirty_region,
	                      region);
	pixman_region32_intersect_rect(&buffer->dirty_region,
	                               &buffer->dirty_region, 0, 0,
	                               base->base.width, base->base.height);
}

void
buffer_flush(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	/*
	 * Called once the renderer targeting this buffer is done with it, which
	 * for a CPU buffer means the CPU has just finished drawing into it. Unless
	 * that renderer said where, it could have been anywhere.
	 */
	if (buffer->cpu && !buffer->upload_only) {
		if (!buffer->damage_reported)
			buffer->dirty = true;
		buffer->damage_reported = false;
	}
}

void
buffer_destroy(struct buffer *base)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);

	if (buffer->texture)
		delete_texture(buffer->context, buffer->texture);
	pixman_region32_fini(&buffer->dirty_region);

	if (buffer->cpu) {
		if (buffer->dumb) {
			struct drm_mode_destroy_dumb destroy_dumb = {
				.handle = buffer->handle
			};

			if (base->base.map)
				munmap(base->base.map, buffer->mapping_size);
			drmIoctl(buffer->context->fd, DRM_IOCTL_MODE_DESTROY_DUMB,
			         &destroy_dumb);
		} else if (buffer->mapping_size) {
			munmap(base->base.map, buffer->mapping_size);
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

/*
 * Copy one box of pixels into the bound texture.
 *
 * GLES2 has no GL_UNPACK_ROW_LENGTH, so without GL_EXT_unpack_subimage the
 * rows of a box narrower than the pitch cannot be uploaded in one call.
 */
static void
upload_box(struct gbm_context *context, const uint8_t *pixels, uint32_t pitch,
           uint32_t buffer_width, const pixman_box32_t *box)
{
	uint32_t row_pixels = pitch / 4;
	int32_t x = box->x1, y = box->y1;
	int32_t width = box->x2 - box->x1, height = box->y2 - box->y1;

	if (width <= 0 || height <= 0)
		return;

	if (row_pixels == (uint32_t)width) {
		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, GL_BGRA_EXT,
		                GL_UNSIGNED_BYTE, pixels + (size_t)y * pitch);
	} else if (context->has_unpack_subimage) {
		glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT, row_pixels);
		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, GL_BGRA_EXT,
		                GL_UNSIGNED_BYTE,
		                pixels + (size_t)y * pitch + (size_t)x * 4);
		glPixelStorei(GL_UNPACK_ROW_LENGTH_EXT, 0);
	} else if (row_pixels == buffer_width) {
		/* Whole rows are contiguous, so widen the box to them. */
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, row_pixels, height,
		                GL_BGRA_EXT, GL_UNSIGNED_BYTE,
		                pixels + (size_t)y * pitch);
	} else {
		int32_t row;

		for (row = 0; row < height; ++row) {
			glTexSubImage2D(GL_TEXTURE_2D, 0, x, y + row, width, 1,
			                GL_BGRA_EXT, GL_UNSIGNED_BYTE,
			                pixels + (size_t)(y + row) * pitch
			                    + (size_t)x * 4);
		}
	}
}

/* Upload every box of a region, or the box around them all past a point. */
static void
upload_region(struct gbm_context *context, const uint8_t *pixels,
              uint32_t pitch, uint32_t buffer_width, pixman_region32_t *region)
{
	pixman_box32_t *boxes;
	int count;

	boxes = pixman_region32_rectangles(region, &count);
	/*
	 * A region fragmented into many small boxes costs more in calls than the
	 * few extra bytes its bounding box would upload.
	 */
	if (count > DIRTY_BOX_LIMIT) {
		boxes = pixman_region32_extents(region);
		count = 1;
	}
	while (count--)
		upload_box(context, pixels, pitch, buffer_width, boxes++);
	++context->submissions;
}

/* Give a CPU buffer's texture its storage, blank, the first time. */
static void
allocate_texture_storage(struct gbm_buffer *buffer)
{
	uint32_t width = buffer->base.base.width;
	uint32_t height = buffer->base.base.height;
	void *zero;

	if (buffer->tex_allocated)
		return;
	/* Blank rather than undefined: the edges of what is uploaded later are
	 * filtered against their neighbours. */
	zero = calloc((size_t)width * 4, height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA_EXT, width, height, 0, GL_BGRA_EXT,
	             GL_UNSIGNED_BYTE, zero);
	free(zero);
	buffer->tex_allocated = true;
}

/* Lazily wrap a buffer's EGLImage in a GL texture. */
static GLuint
buffer_texture(struct gbm_buffer *buffer)
{
	struct gbm_context *context = buffer->context;
	bool fresh = buffer->texture == 0;

	if (fresh) {
		if (!buffer->cpu && buffer->image == EGL_NO_IMAGE_KHR)
			return 0;

		glGenTextures(1, &buffer->texture);
		bind_texture_for_upload(context, buffer->texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		if (!buffer->cpu) {
			context->image_target_texture_2d(GL_TEXTURE_2D, buffer->image);
			return buffer->texture;
		}
	}

	/* Nothing to bring across: the pixels only ever arrive by upload. */
	if (buffer->upload_only) {
		if (!buffer->tex_allocated) {
			bind_texture_for_upload(context, buffer->texture);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
			allocate_texture_storage(buffer);
		}
		return buffer->texture;
	}

	/* CPU-backed contents live in system memory and must be uploaded. */
	if (buffer->cpu && (fresh || buffer->dirty
	                    || pixman_region32_not_empty(&buffer->dirty_region))) {
		if (!buffer->base.base.map)
			return 0;

		bind_texture_for_upload(context, buffer->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		/* Allocate storage once; refreshes are sub-image updates. */
		allocate_texture_storage(buffer);

		if (fresh || buffer->dirty) {
			pixman_region32_t full;

			pixman_region32_init_rect(&full, 0, 0, buffer->base.base.width,
			                          buffer->base.base.height);
			upload_region(context, buffer->base.base.map,
			              buffer->base.base.pitch, buffer->base.base.width,
			              &full);
			pixman_region32_fini(&full);
		} else {
			upload_region(context, buffer->base.base.map,
			              buffer->base.base.pitch, buffer->base.base.width,
			              &buffer->dirty_region);
		}

		buffer->dirty = false;
		pixman_region32_clear(&buffer->dirty_region);
	}

	return buffer->texture;
}

/*
 * Copy a region of the caller's pixels into the texture, straight from where
 * they are. This is how a client's shared memory reaches the GPU: one copy,
 * by the driver, rather than a copy into a buffer of ours and another out
 * of it.
 */
bool
buffer_upload(struct buffer *base, const void *pixels, uint32_t pitch,
              pixman_region32_t *region)
{
	struct gbm_buffer *buffer = gbm_buffer(&base->base);
	struct gbm_context *context = buffer->context;
	pixman_region32_t clipped;

	/* Rows have to start on a 4-byte boundary for the unpack alignment,
	 * and only 32-bit formats are uploaded here. */
	if (!buffer->cpu || !pixels || pitch % 4 || pitch < 4)
		return false;
	if (!buffer_texture(buffer))
		return false;

	/* The caller's rows may be narrower than the buffer, which is padded
	 * out past what a client hands over; only what both hold is copied. */
	pixman_region32_init(&clipped);
	pixman_region32_intersect_rect(&clipped, region, 0, 0,
	                               base->base.width < pitch / 4 ? base->base.width : pitch / 4,
	                               base->base.height);
	if (pixman_region32_not_empty(&clipped)) {
		bind_texture_for_upload(context, buffer->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		allocate_texture_storage(buffer);
		upload_region(context, pixels, pitch, base->base.width, &clipped);
	}
	pixman_region32_fini(&clipped);
	return true;
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

	renderer->solid.proj_valid = false;
	renderer->textured.proj_valid = false;
	renderer->glyph.proj_valid = false;
}

/*
 * Make this renderer's framebuffer the one drawn to, and its clip the
 * scissor. Renderers share the GL context, so another one may have bound its
 * own since our target was set. Either change draws the staged quads first,
 * since they were staged for the state being replaced.
 */
static bool
bind_target(struct gles_renderer *renderer)
{
	struct gbm_context *context = renderer->context;

	if (!renderer->target_texture)
		return false;

	if (context->gl.fbo != renderer->fbo) {
		batch_flush(context);
		glBindFramebuffer(GL_FRAMEBUFFER, renderer->fbo);
		glViewport(0, 0, renderer->target_width, renderer->target_height);
		context->gl.fbo = renderer->fbo;
	}

	if (context->gl.scissor != (int)renderer->clip_enabled
	    || (renderer->clip_enabled
	        && memcmp(&context->gl.scissor_box, &renderer->clip,
	                  sizeof renderer->clip) != 0)) {
		batch_flush(context);
		if (renderer->clip_enabled) {
			glEnable(GL_SCISSOR_TEST);
			glScissor(renderer->clip.x1, renderer->clip.y1,
			          renderer->clip.x2 - renderer->clip.x1,
			          renderer->clip.y2 - renderer->clip.y1);
			context->gl.scissor_box = renderer->clip;
		} else {
			glDisable(GL_SCISSOR_TEST);
		}
		context->gl.scissor = renderer->clip_enabled;
	}

	return true;
}

static void
load_projection(struct gles_renderer *renderer, GLint location, bool *valid)
{
	if (!*valid) {
		glUniformMatrix4fv(location, 1, GL_FALSE, renderer->proj);
		*valid = true;
	}
}

/*
 * Bind the textured program to `src_base` and set blending up the way both the
 * region and the scaled path want it. False when there is no target or the
 * source has no texture to draw from.
 */
static bool
setup_textured(struct gles_renderer *renderer, struct buffer *src_base,
               bool blend)
{
	struct gbm_context *context = renderer->context;
	struct gbm_buffer *src = gbm_buffer(&src_base->base);
	GLuint texture;
	bool opaque;

	if (!bind_target(renderer))
		return false;
	if (!(texture = buffer_texture(src)))
		return false;

	/* An XRGB source carries no meaningful alpha, so force it opaque. The
	 * uniform applies to whatever is staged too, so that is drawn first. */
	opaque = src_base->base.format == WLD_FORMAT_XRGB8888;
	if (renderer->textured.opaque != opaque)
		batch_flush(context);

	/* wld buffers hold premultiplied alpha. */
	batch_begin(renderer, renderer->textured.program, texture, blend);
	load_projection(renderer, renderer->textured.proj,
	                &renderer->textured.proj_valid);

	if (renderer->textured.opaque != opaque) {
		GLfloat mul[4] = { 1.0f, 1.0f, 1.0f, opaque ? 0.0f : 1.0f };
		GLfloat add[4] = { 0.0f, 0.0f, 0.0f, opaque ? 1.0f : 0.0f };

		glUniform4fv(renderer->textured.mul, 1, mul);
		glUniform4fv(renderer->textured.add, 1, add);
		renderer->textured.opaque = opaque;
	}

	return true;
}

/* Solid fills sample nothing, so whatever texture is bound stays bound. */
static void
setup_solid(struct gles_renderer *renderer)
{
	struct gbm_context *context = renderer->context;

	batch_begin(renderer, renderer->solid.program, context->gl.texture, false);
	load_projection(renderer, renderer->solid.proj,
	                &renderer->solid.proj_valid);
}

static const GLfloat white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

static void
composite_region(struct wld_renderer *base, struct buffer *src_base,
                 int32_t dst_x, int32_t dst_y, pixman_region32_t *region,
                 bool blend)
{
	struct gles_renderer *renderer = gles_renderer(base);
	GLfloat src_w = (GLfloat)src_base->base.width;
	GLfloat src_h = (GLfloat)src_base->base.height;
	pixman_box32_t *boxes;
	GLfloat *v;
	int count, i;

	boxes = pixman_region32_rectangles(region, &count);
	if (count <= 0)
		return;
	if (!setup_textured(renderer, src_base, blend))
		return;
	if (!(v = batch_reserve(renderer->context, count)))
		return;

	for (i = 0; i < count; ++i) {
		v = put_quad(v, boxes[i].x1 + dst_x, boxes[i].y1 + dst_y,
		             boxes[i].x2 + dst_x, boxes[i].y2 + dst_y,
		             boxes[i].x1 / src_w, boxes[i].y1 / src_h,
		             boxes[i].x2 / src_w, boxes[i].y2 / src_h, white);
	}
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
	struct gbm_context *context = renderer->context;
	GLuint texture;

	if (!buffer) {
		batch_flush(context);
		renderer->target_texture = 0;
		renderer->clip_enabled = false;
		return true;
	}

	if (buffer->base.impl != &wld_buffer_impl)
		return false;

	if (!(texture = buffer_texture(gbm_buffer(&buffer->base))))
		return false;

	/* The staged quads were for the target being replaced. */
	batch_flush(context);
	renderer->clip_enabled = false;

	/*
	 * Already attached and bound: nothing to do but the bookkeeping below.
	 * Attaching again would also mean asking the driver to validate the
	 * framebuffer again, which is not free on every driver.
	 */
	if (renderer->target_texture != texture || context->gl.fbo != renderer->fbo) {
		glBindFramebuffer(GL_FRAMEBUFFER, renderer->fbo);
		context->gl.fbo = renderer->fbo;
		if (renderer->target_texture != texture) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			                       GL_TEXTURE_2D, texture, 0);

			if (glCheckFramebufferStatus(GL_FRAMEBUFFER)
			    != GL_FRAMEBUFFER_COMPLETE) {
				DEBUG("framebuffer incomplete for target buffer\n");
				renderer->target_texture = 0;
				return false;
			}
		}
		glViewport(0, 0, buffer->base.width, buffer->base.height);
	}

	renderer->target_texture = texture;
	if (renderer->target_width != buffer->base.width
	    || renderer->target_height != buffer->base.height
	    || !renderer->proj[15]) {
		renderer->target_width = buffer->base.width;
		renderer->target_height = buffer->base.height;
		set_projection(renderer, buffer->base.width, buffer->base.height);
	}

	return true;
}

void
renderer_fill_rectangle(struct wld_renderer *base, uint32_t color, int32_t x,
                        int32_t y, uint32_t width, uint32_t height)
{
	struct gles_renderer *renderer = gles_renderer(base);
	GLfloat rgba[4], *v;

	if (!bind_target(renderer))
		return;
	setup_solid(renderer);
	if (!(v = batch_reserve(renderer->context, 1)))
		return;

	color_to_gl(color, rgba);
	put_quad(v, x, y, x + (int32_t)width, y + (int32_t)height, 0, 0, 0, 0,
	         rgba);
}

void
renderer_fill_region(struct wld_renderer *base, uint32_t color,
                     pixman_region32_t *region)
{
	struct gles_renderer *renderer = gles_renderer(base);
	pixman_box32_t *boxes;
	GLfloat rgba[4], *v;
	int count, i;

	boxes = pixman_region32_rectangles(region, &count);
	if (count <= 0 || !bind_target(renderer))
		return;
	setup_solid(renderer);
	if (!(v = batch_reserve(renderer->context, count)))
		return;

	color_to_gl(color, rgba);
	for (i = 0; i < count; ++i) {
		v = put_quad(v, boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2,
		             0, 0, 0, 0, rgba);
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

void
renderer_blend_scaled(struct wld_renderer *base, struct buffer *src_base,
                      const struct wld_rect *dst, const struct wld_frect *src)
{
	struct gles_renderer *renderer = gles_renderer(base);
	GLfloat src_w = (GLfloat)src_base->base.width;
	GLfloat src_h = (GLfloat)src_base->base.height;
	GLfloat *v;

	if (src_w <= 0.0f || src_h <= 0.0f)
		return;
	if (!setup_textured(renderer, src_base, true))
		return;
	if (!(v = batch_reserve(renderer->context, 1)))
		return;

	/*
	 * One quad, with the source rectangle's edges as texture coordinates and
	 * the destination's as vertex positions. The sampler does the scaling,
	 * and because texture coordinates are interpolated to fragment centres it
	 * does the half-texel bookkeeping that the pixman path has to spell out
	 * in scaled_transform() for itself.
	 *
	 * buffer_texture() already asks for GL_LINEAR in both directions, so
	 * there is no filter state to set here.
	 */
	put_quad(v, dst->x, dst->y, dst->x + (int32_t)dst->width,
	         dst->y + (int32_t)dst->height,
	         (GLfloat)(src->x / src_w), (GLfloat)(src->y / src_h),
	         (GLfloat)((src->x + src->width) / src_w),
	         (GLfloat)((src->y + src->height) / src_h), white);
}

/**** Glyph atlas ****/

static bool
atlas_create(struct gbm_context *context)
{
	struct glyph_atlas *atlas = &context->atlas;
	GLint max_size = 0;
	uint8_t *zero;

	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_size);
	atlas->size = GLYPH_ATLAS_SIZE;
	if (max_size > 0 && (GLint)atlas->size > max_size)
		atlas->size = max_size;

	if (!(atlas->slots = calloc(GLYPH_TABLE_SIZE, sizeof *atlas->slots)))
		return false;
	/* Start blank, so padding never samples undefined texels. */
	if (!(zero = calloc(atlas->size, atlas->size))) {
		free(atlas->slots);
		atlas->slots = NULL;
		return false;
	}

	glGenTextures(1, &atlas->texture);
	bind_texture_for_upload(context, atlas->texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, atlas->size, atlas->size, 0,
	             GL_ALPHA, GL_UNSIGNED_BYTE, zero);
	free(zero);

	return true;
}

/*
 * Forget every glyph. Draws already issued keep what they sampled: GL orders
 * the uploads that later overwrite the texels after them.
 */
static void
atlas_reset(struct glyph_atlas *atlas)
{
	memset(atlas->slots, 0, GLYPH_TABLE_SIZE * sizeof *atlas->slots);
	atlas->count = 0;
	atlas->shelf_x = atlas->shelf_y = atlas->shelf_height = 0;
}

enum glyph_result {
	GLYPH_OK,
	GLYPH_SKIP, /* nothing to draw for this glyph */
	GLYPH_FULL, /* reset the atlas and ask again */
};

static uint8_t *
glyph_alpha(const FT_Bitmap *bitmap)
{
	uint8_t *pixels;
	uint32_t row, col;

	if (bitmap->pixel_mode != FT_PIXEL_MODE_GRAY
	    && bitmap->pixel_mode != FT_PIXEL_MODE_MONO)
		return NULL;
	if (!(pixels = malloc((size_t)bitmap->width * bitmap->rows)))
		return NULL;

	for (row = 0; row < bitmap->rows; ++row) {
		const uint8_t *src = bitmap->buffer + (ptrdiff_t)row * bitmap->pitch;
		uint8_t *dst = pixels + (size_t)row * bitmap->width;

		if (bitmap->pixel_mode == FT_PIXEL_MODE_GRAY) {
			memcpy(dst, src, bitmap->width);
			continue;
		}
		for (col = 0; col < bitmap->width; ++col)
			dst[col] = (src[col >> 3] & (0x80 >> (col & 7))) ? 0xff : 0x00;
	}

	return pixels;
}

/*
 * Find a glyph in the atlas, uploading it on a miss. Glyph serials are unique
 * for the life of the process, so a font closed and another loaded at the same
 * address can never hit a stale entry.
 */
static enum glyph_result
atlas_glyph(struct gbm_context *context, struct glyph *glyph,
            const struct glyph_slot **out)
{
	struct glyph_atlas *atlas = &context->atlas;
	const FT_Bitmap *bitmap = &glyph->bitmap;
	uint32_t width = bitmap->width, height = bitmap->rows;
	struct glyph_slot *slot;
	uint8_t *pixels;
	size_t index;

	if (width == 0 || height == 0)
		return GLYPH_SKIP;
	if (!atlas->slots && !atlas_create(context))
		return GLYPH_SKIP;
	/* Larger than the whole atlas: no amount of eviction makes room. */
	if (width + GLYPH_PADDING > atlas->size
	    || height + GLYPH_PADDING > atlas->size)
		return GLYPH_SKIP;

	index = (size_t)((glyph->serial * UINT64_C(0x9e3779b97f4a7c15)) >> 52)
	        & (GLYPH_TABLE_SIZE - 1);
	for (; atlas->slots[index].serial;
	     index = (index + 1) & (GLYPH_TABLE_SIZE - 1)) {
		if (atlas->slots[index].serial == glyph->serial) {
			*out = &atlas->slots[index];
			return GLYPH_OK;
		}
	}

	if (atlas->count >= GLYPH_TABLE_LOAD)
		return GLYPH_FULL;
	if (atlas->shelf_x + width + GLYPH_PADDING > atlas->size) {
		atlas->shelf_y += atlas->shelf_height;
		atlas->shelf_x = 0;
		atlas->shelf_height = 0;
	}
	if (atlas->shelf_y + height + GLYPH_PADDING > atlas->size)
		return GLYPH_FULL;

	if (!(pixels = glyph_alpha(bitmap)))
		return GLYPH_SKIP;

	bind_texture_for_upload(context, atlas->texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_2D, 0, atlas->shelf_x, atlas->shelf_y, width,
	                height, GL_ALPHA, GL_UNSIGNED_BYTE, pixels);
	free(pixels);
	++context->submissions;

	slot = &atlas->slots[index];
	slot->serial = glyph->serial;
	slot->x = atlas->shelf_x;
	slot->y = atlas->shelf_y;
	slot->width = width;
	slot->height = height;
	++atlas->count;

	atlas->shelf_x += width + GLYPH_PADDING;
	if (height + GLYPH_PADDING > atlas->shelf_height)
		atlas->shelf_height = height + GLYPH_PADDING;

	*out = slot;
	return GLYPH_OK;
}

void
renderer_draw_text(struct wld_renderer *base, struct font *font, uint32_t color,
                   int32_t x, int32_t y, const char *text, uint32_t length,
                   struct wld_extents *extents)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct gbm_context *context = renderer->context;
	const struct glyph_slot *slot;
	enum glyph_result result;
	struct glyph *glyph;
	FT_UInt glyph_index;
	GLfloat rgba[4], scale;
	uint32_t origin_x = 0, c;
	GLfloat *v;
	int ret;

	if (length == -1)
		length = strlen(text);

	/* The advance is still owed to the caller when there is nothing to draw
	 * into, so a missing target only suppresses the drawing. */
	if (!bind_target(renderer)
	    || (!context->atlas.slots && !atlas_create(context))) {
		while ((ret = FcUtf8ToUcs4((FcChar8 *)text, &c, length)) > 0
		       && c != '\0') {
			text += ret;
			length -= ret;
			glyph_index = FT_Get_Char_Index(font->face, c);
			if (font_ensure_glyph(font, glyph_index))
				origin_x += font->glyphs[glyph_index]->advance;
		}
		goto done;
	}

	scale = 1.0f / (GLfloat)context->atlas.size;

	color_to_gl(color, rgba);

	while ((ret = FcUtf8ToUcs4((FcChar8 *)text, &c, length)) > 0 && c != '\0') {
		text += ret;
		length -= ret;
		glyph_index = FT_Get_Char_Index(font->face, c);

		if (!font_ensure_glyph(font, glyph_index))
			continue;

		glyph = font->glyphs[glyph_index];
		/* A missing glyph is uploaded, which draws what is staged first:
		 * anything pointing into the atlas has to go before it is reused. */
		result = atlas_glyph(context, glyph, &slot);
		if (result == GLYPH_FULL) {
			batch_flush(context);
			atlas_reset(&context->atlas);
			result = atlas_glyph(context, glyph, &slot);
		}

		if (result == GLYPH_OK) {
			GLfloat gx = x + (int32_t)origin_x + glyph->x;
			GLfloat gy = y + glyph->y;

			batch_begin(renderer, renderer->glyph.program,
			            context->atlas.texture, true);
			load_projection(renderer, renderer->glyph.proj,
			                &renderer->glyph.proj_valid);
			if ((v = batch_reserve(context, 1))) {
				put_quad(v, gx, gy, gx + slot->width, gy + slot->height,
				         slot->x * scale, slot->y * scale,
				         (slot->x + slot->width) * scale,
				         (slot->y + slot->height) * scale, rgba);
			}
		}

		origin_x += glyph->advance;
	}

done:
	if (extents)
		extents->advance = origin_x;
}

bool
renderer_read_pixels(struct wld_renderer *base, int32_t x, int32_t y,
                     uint32_t width, uint32_t height, uint32_t pitch,
                     void *data)
{
	struct gles_renderer *renderer = gles_renderer(base);
	uint64_t row_bytes = (uint64_t)width * 4;

	if (!bind_target(renderer) || !data || x < 0 || y < 0 ||
	    width == 0 || height == 0 || pitch < row_bytes ||
	    (uint64_t)x + width > renderer->target_width ||
	    (uint64_t)y + height > renderer->target_height)
		return false;

	/* Client-memory glReadPixels completes the read before returning, and
	 * it reads what has been drawn, so the batch has to be. */
	batch_flush(renderer->context);

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

/*
 * How this driver behaves around fence synchronization, learned on the first
 * fence rather than assumed.
 *
 * Two things vary, and the NVIDIA driver gets both wrong in ways that matter:
 *
 *  - Ownership of the descriptor handed to eglCreateSyncKHR.
 *    EGL_ANDROID_native_fence_sync says the implementation takes it when the
 *    call succeeds and closes it with the sync object. Closing it ourselves
 *    against a driver that does that is a double close of a descriptor number
 *    the driver may have reused, so we have to know rather than guess.
 *
 *  - Whether the GPU-side wait leaks. On this driver eglWaitSyncKHR leaks one
 *    sync_file descriptor per call, permanently: create and destroy alone
 *    balance, and adding the wait costs exactly one descriptor every time.
 *    At one synchronized commit per client frame that exhausts a compositor's
 *    descriptor table in minutes, and everything downstream of a descriptor
 *    then fails at once -- clients cannot send buffers, and this function
 *    stops working, which removes the very wait it exists to perform.
 *
 * So run one full create/wait/destroy cycle with both questions instrumented,
 * and act on the answers from then on. Where the wait leaks, only report
 * whether the fence has already signalled: a sync_file becomes readable when
 * its fence signals, so the caller can wait for exactly that event in its own
 * event loop instead of blocking here.
 */
enum fence_fd_ownership {
	FENCE_FD_UNKNOWN,
	FENCE_FD_CONSUMED, /* the driver closed it; it is not ours to close */
	FENCE_FD_RETAINED, /* the driver left it open; we still own it */
};

static enum fence_fd_ownership fence_fd_ownership;
static bool gpu_wait_leaks;
static bool fence_behavior_known;

/*
 * How many descriptors the process has open, or -1 where that cannot be asked.
 *
 * Counting is the only reliable way to see this leak: the descriptor the
 * driver keeps lands in the slot the descriptor it consumed just vacated, so
 * the lowest free number does not move even though one was kept.
 */
static int
count_open_fds(void)
{
#ifdef __linux__
	DIR *dir = opendir("/proc/self/fd");
	struct dirent *entry;
	int total = 0;

	if (!dir)
		return -1;
	while ((entry = readdir(dir))) {
		if (entry->d_name[0] != '.')
			++total;
	}
	closedir(dir);
	/* The walk listed the descriptor it was walking with. */
	return total > 0 ? total - 1 : 0;
#else
	return -1;
#endif
}

static bool
same_open_file(int fd, const struct stat *before)
{
	struct stat now;

	return fstat(fd, &now) == 0 && now.st_dev == before->st_dev
	       && now.st_ino == before->st_ino;
}

/* A sync_file signals by becoming readable. This only looks; it never waits. */
static bool
fence_signalled(int fence_fd)
{
	struct pollfd pollfd = {.fd = fence_fd, .events = POLLIN};
	int ret;

	do {
		ret = poll(&pollfd, 1, 0);
	} while (ret < 0 && errno == EINTR);

	return ret > 0;
}

bool
renderer_wait_fence(struct wld_renderer *base, int fence_fd)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct gbm_context *context = renderer->context;
	EGLSyncKHR sync;
	EGLint attribs[3];
	struct stat identity;
	bool probing, probe_identity, waited;
	int dup_fd, open_before = -1;

	if (!context->has_fence_sync)
		return false;

	/* A negative descriptor only asks whether fences are supported. */
	if (fence_fd < 0)
		return true;

	if (gpu_wait_leaks)
		return fence_signalled(fence_fd);

	probing = !fence_behavior_known;
	/* Counted before our own duplicate exists, and compared after it is gone
	 * again, so only what the driver kept is left in the difference. */
	if (probing)
		open_before = count_open_fds();

	/* The caller keeps its own descriptor whatever the driver does with ours. */
	if ((dup_fd = fcntl(fence_fd, F_DUPFD_CLOEXEC, 0)) < 0)
		return false;

	probe_identity = probing && fstat(dup_fd, &identity) == 0;

	attribs[0] = EGL_SYNC_NATIVE_FENCE_FD_ANDROID;
	attribs[1] = dup_fd;
	attribs[2] = EGL_NONE;

	sync = context->create_sync(context->display,
	                            EGL_SYNC_NATIVE_FENCE_ANDROID, attribs);
	if (sync == EGL_NO_SYNC_KHR) {
		/* Ownership only ever transfers on success, so this one is ours. */
		close(dup_fd);
		return false;
	}

	/*
	 * Wait on the GPU rather than the CPU where that works: it only orders the
	 * commands queued after it, and never blocks the caller on a client that
	 * is slow to finish drawing.
	 */
	waited = context->wait_sync(context->display, sync, 0);
	context->destroy_sync(context->display, sync);

	if (probing) {
		int open_after;

		if (probe_identity) {
			fence_fd_ownership = same_open_file(dup_fd, &identity)
			                         ? FENCE_FD_RETAINED
			                         : FENCE_FD_CONSUMED;
		}
		/* Take our own descriptor back first, so it is not mistaken for one
		 * the driver kept. */
		if (fence_fd_ownership == FENCE_FD_RETAINED)
			close(dup_fd);

		open_after = count_open_fds();
		if (open_before >= 0 && open_after > open_before) {
			gpu_wait_leaks = true;
			fprintf(stderr,
			        "wld: this driver leaks a descriptor per GPU fence wait; "
			        "waiting on fences directly instead\n");
		}
		/* Both answers come from this one cycle; do not probe again. */
		if (probe_identity)
			fence_behavior_known = true;
		return waited;
	}

	/* Unknown only when the probe itself failed: leave the descriptor alone. */
	if (fence_fd_ownership == FENCE_FD_RETAINED)
		close(dup_fd);

	return waited;
}

int
renderer_export_fence(struct wld_renderer *base)
{
	struct gbm_context *context = gles_renderer(base)->context;
	EGLint attribs[] = {
		EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID,
		EGL_NONE
	};
	EGLSyncKHR sync;
	int fd;

	batch_flush(context);
	if (context->has_native_fence) {
		/*
		 * Nothing has been queued since the last fence was made, so it still
		 * says exactly when everything queued is done. Every buffer a client
		 * replaces asks for one, and a frame's worth of them can share.
		 */
		if (context->fence_fd >= 0
		    && context->fence_submissions == context->submissions) {
			fd = fcntl(context->fence_fd, F_DUPFD_CLOEXEC, 0);
			if (fd >= 0)
				return fd;
		}
		sync = context->create_sync(context->display,
		                            EGL_SYNC_NATIVE_FENCE_ANDROID, attribs);
		if (sync != EGL_NO_SYNC_KHR) {
			/* The fence only gets a descriptor once it has been flushed. */
			glFlush();
			fd = context->dup_native_fence_fd(context->display, sync);
			context->destroy_sync(context->display, sync);
			if (fd >= 0) {
				int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);

				if (context->fence_fd >= 0)
					close(context->fence_fd);
				context->fence_fd = fd;
				context->fence_submissions = context->submissions;
				if (dup_fd >= 0)
					return dup_fd;
			}
		}
	}

	glFinish();
	context->unfinished = false;
	return -1;
}

void
renderer_flush(struct wld_renderer *base)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct gbm_context *context = renderer->context;

	/*
	 * Submitting the work is enough. A scanout buffer goes to KMS, and
	 * whatever hands it over waits on wld_export_fence() first; a buffer
	 * drawn only to be drawn from again is ordered after this by the GPU
	 * itself. Whatever is read by the CPU catches up where it is read:
	 * mapping a buffer or exporting it finishes first, and reading pixels
	 * back waits on its own. Finishing here would instead stall the caller
	 * for every frame's worth of work so far -- in the middle of a frame,
	 * when it is a titlebar being repainted.
	 */
	batch_flush(context);
	glFlush();
	context->unfinished = true;
}

void
renderer_set_clip(struct wld_renderer *base, const pixman_box32_t *box)
{
	struct gles_renderer *renderer = gles_renderer(base);

	/* Takes effect at the next draw, which draws the staged quads first. */
	renderer->clip_enabled = box != NULL;
	if (box)
		renderer->clip = *box;
}

void
renderer_destroy(struct wld_renderer *base)
{
	struct gles_renderer *renderer = gles_renderer(base);
	struct gbm_context *context = renderer->context;
	struct gles_renderer **link;

	if (context->batch.renderer == renderer) {
		batch_flush(context);
		context->batch.renderer = NULL;
	}
	for (link = &context->renderers; *link; link = &(*link)->next) {
		if (*link == renderer) {
			*link = renderer->next;
			break;
		}
	}

	/* A deleted name can come back from the next glCreateProgram or
	 * glGenFramebuffers, so the cache must not keep claiming it is bound. */
	if (context->gl.program == renderer->solid.program
	    || context->gl.program == renderer->textured.program
	    || context->gl.program == renderer->glyph.program) {
		glUseProgram(0);
		context->gl.program = 0;
	}
	if (context->gl.fbo == renderer->fbo) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		context->gl.fbo = 0;
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
	renderer->glyph.program = link_program(vertex_glyph_src, fragment_glyph_src);

	if (!renderer->solid.program || !renderer->textured.program
	    || !renderer->glyph.program) {
		goto error;
	}

	renderer->solid.proj = glGetUniformLocation(renderer->solid.program, "proj");

	renderer->textured.proj =
	    glGetUniformLocation(renderer->textured.program, "proj");
	renderer->textured.mul =
	    glGetUniformLocation(renderer->textured.program, "mul");
	renderer->textured.add =
	    glGetUniformLocation(renderer->textured.program, "add");
	renderer->textured.opaque = -1;

	renderer->glyph.proj =
	    glGetUniformLocation(renderer->glyph.program, "proj");

	/* Both texturing programs only ever sample unit 0. */
	use_program(context, renderer->textured.program);
	glUniform1i(glGetUniformLocation(renderer->textured.program, "tex"), 0);
	use_program(context, renderer->glyph.program);
	glUniform1i(glGetUniformLocation(renderer->glyph.program, "tex"), 0);

	glGenFramebuffers(1, &renderer->fbo);

	renderer_initialize(&renderer->base, &wld_renderer_impl);
	renderer->next = context->renderers;
	context->renderers = renderer;

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
