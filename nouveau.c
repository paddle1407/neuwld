/* wld: nouveau.c
 *
 * Copyright (c) 2013, 2014 Michael Forney
 *
 * Based in part upon nvc0_exa.c from xf86-video-nouveau, which is:
 *
 *     Copyright 2007 NVIDIA, Corporation
 *     Copyright 2008 Ben Skeggs
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
#include "nouveau/g80_2d.xml.h"
#include "nouveau/g80_defs.xml.h"
#include "nouveau/nv_object.xml.h"
#include "pixman.h"

#include <nouveau.h>
#include <string.h>
#include <sys/mman.h>

enum nv_architecture {
	NV_ARCH_50 = 0x50,
	NV_ARCH_C0 = 0xc0,
	NV_ARCH_E0 = 0xe0
};

struct nouveau_context {
	struct wld_context base;
	struct nouveau_device *device;
	struct nouveau_client *client;
	enum nv_architecture architecture;
};

struct nouveau_renderer {
	struct wld_renderer base;
	struct wld_context *context;
	struct nouveau_object *channel;
	struct nouveau_pushbuf *pushbuf;
	struct nouveau_bufctx *bufctx;
	struct nouveau_object *nvc0_2d;

	struct nouveau_buffer *target;
};

struct nouveau_buffer {
	struct buffer base;
	struct wld_exporter exporter;
	struct nouveau_context *context;
	struct nouveau_bo *bo;
};

#include "interface/buffer.h"
#include "interface/context.h"
#include "interface/renderer.h"
#define DRM_DRIVER_NAME nouveau
#define DRM_DRIVER_REQUIRES_PCI true
#include "interface/drm.h"
IMPL(nouveau_context, wld_context)
IMPL(nouveau_renderer, wld_renderer)
IMPL(nouveau_buffer, wld_buffer)

static void
pack_gray_row_to_mono(uint8_t *dst, const uint8_t *src, uint32_t width)
{
	uint32_t x, bytes_per_row;

	bytes_per_row = (width + 7) / 8;
	memset(dst, 0, bytes_per_row);

	for (x = 0; x < width; ++x) {
		if (src[x] >= 128)
			dst[x / 8] |= 0x80 >> (x & 7);
	}
}

/**** DRM driver ****/
bool
driver_device_supported(uint32_t vendor_id, uint32_t device_id)
{
	return vendor_id == 0x10de;
}

struct wld_context *
driver_create_context(int drm_fd)
{
	struct nouveau_context *context;

	if (!(context = malloc(sizeof *context)))
		goto error0;

	if (nouveau_device_wrap(drm_fd, 0, &context->device) != 0)
		goto error1;

	switch (context->device->chipset & ~0xf) {
	/* TODO: Support NV50
	case 0x50:
	case 0x80:
	case 0x90:
	case 0xa0:
		context->architecture = NV_ARCH_50;
		break;
	*/
	case 0xc0:
	case 0xd0:
		context->architecture = NV_ARCH_C0;
		break;
	/* TODO: Support NVE0
	case 0xe0:
	case 0xf0:
	case 0x100:
		context->architecture = NV_ARCH_E0;
		break;
	*/
	default:
		goto error2;
	}

	if (nouveau_client_new(context->device, &context->client) != 0)
		goto error2;

	context_initialize(&context->base, &wld_context_impl);

	return &context->base;

error2:
	nouveau_device_del(&context->device);
error1:
	free(context);
error0:
	return NULL;
}

/**** Context ****/
static inline bool
ensure_space(struct nouveau_pushbuf *push, uint32_t count)
{
	if (push->end - push->cur > count)
		return true;

	return nouveau_pushbuf_space(push, count, 0, 0) == 0;
}

static inline void
nv_add_dword(struct nouveau_pushbuf *push, uint32_t dword)
{
	*push->cur++ = dword;
}

static inline void
nv_add_data(struct nouveau_pushbuf *push, const void *data, size_t bytes)
{
	size_t count = (bytes + 3) / 4;
	memset(push->cur, 0, count * 4);
	memcpy(push->cur, data, bytes);
	push->cur += count;
}

static inline uint32_t
nvc0_format(uint32_t format)
{
	switch (format) {
	case WLD_FORMAT_XRGB8888:
		return G80_SURFACE_FORMAT_BGRX8_UNORM;
	case WLD_FORMAT_ARGB8888:
		return G80_SURFACE_FORMAT_BGRA8_UNORM;
	}

	return 0;
}

enum {
	GF100_COMMAND_TYPE_INCREASING = 1,
	GF100_COMMAND_TYPE_NON_INCREASING = 3,
	GF100_COMMAND_TYPE_INLINE = 4
};

enum {
	GF100_SUBCHANNEL_2D = 3,
};

static inline uint32_t
nvc0_command(uint8_t type, uint8_t subchannel, uint16_t method, uint16_t count_or_value)
{
	return type << 29 | count_or_value << 16 | subchannel << 13 | method >> 2;
}

static inline void
nvc0_inline(struct nouveau_pushbuf *push, uint8_t subchannel, uint16_t method, uint16_t value)
{
	nv_add_dword(push, nvc0_command(GF100_COMMAND_TYPE_INLINE,
	                                subchannel, method, value));
}

static inline void
nvc0_methods_array(struct nouveau_pushbuf *push, uint16_t method,
                   uint16_t count, const uint32_t *dwords)
{
	nv_add_dword(push, nvc0_command(GF100_COMMAND_TYPE_INCREASING,
	                               GF100_SUBCHANNEL_2D, method, count));
	while (count--)
		nv_add_dword(push, *dwords++);
}

#define nvc0_2d(push, method, count, ...) \
	nvc0_methods_array((push), (method), (count), (const uint32_t[]){__VA_ARGS__})
#define nvc0_2d_inline(push, method, value) \
	nvc0_inline(push, GF100_SUBCHANNEL_2D, method, value)

static bool
nvc0_2d_initialize(struct nouveau_renderer *renderer)
{
	int ret;

	ret = nouveau_object_new(renderer->channel, GF100_2D, GF100_2D, NULL, 0,
	                         &renderer->nvc0_2d);

	if (ret != 0)
		goto error0;

	if (!ensure_space(renderer->pushbuf, 5))
		goto error1;

	nvc0_2d(renderer->pushbuf, NV1_SUBCHAN_OBJECT, 1,
	        renderer->nvc0_2d->handle);
	nvc0_2d_inline(renderer->pushbuf, G80_2D_OPERATION,
	               G80_2D_OPERATION_SRCCOPY_AND);
	nvc0_2d_inline(renderer->pushbuf, G80_2D_UNK0884, 0x3f);
	nvc0_2d_inline(renderer->pushbuf, G80_2D_UNK0888, 1);

	return true;

error1:
	nouveau_object_del(&renderer->nvc0_2d);
error0:
	return false;
}

static void
nvc0_2d_finalize(struct nouveau_renderer *renderer)
{
	nouveau_object_del(&renderer->nvc0_2d);
}

struct wld_renderer *
context_create_renderer(struct wld_context *base)
{
	struct nouveau_context *context = nouveau_context(base);
	struct nouveau_renderer *renderer;
	struct nvc0_fifo fifo = {};
	int ret;

	if (!(renderer = malloc(sizeof *renderer)))
		goto error0;

	ret = nouveau_object_new(&context->device->object, 0,
	                         NOUVEAU_FIFO_CHANNEL_CLASS, &fifo, sizeof fifo,
	                         &renderer->channel);

	if (ret != 0)
		goto error1;

	ret = nouveau_pushbuf_new(context->client, renderer->channel, 4, 32 * 1024,
	                          true, &renderer->pushbuf);

	if (ret != 0)
		goto error2;

	if (nouveau_bufctx_new(context->client, 1, &renderer->bufctx) != 0)
		goto error3;

	if (!nvc0_2d_initialize(renderer))
		goto error4;

	renderer_initialize(&renderer->base, &wld_renderer_impl);
	renderer->context = base;
	renderer->target = NULL;

	return &renderer->base;

error4:
	nouveau_bufctx_del(&renderer->bufctx);
error3:
	nouveau_pushbuf_del(&renderer->pushbuf);
error2:
	nouveau_object_del(&renderer->channel);
error1:
	free(renderer);
error0:
	return NULL;
}

static bool export(struct wld_exporter *exporter, struct wld_buffer *base,
                   uint32_t type, union wld_object *object)
{
	struct nouveau_buffer *buffer = nouveau_buffer(base);

	switch (type) {
	case WLD_DRM_OBJECT_HANDLE:
		object->u32 = buffer->bo->handle;
		return true;
	case WLD_DRM_OBJECT_PRIME_FD:
		if (nouveau_bo_set_prime(buffer->bo, &object->i) != 0)
			return false;
		return true;
	default:
		return false;
	}
}

static struct nouveau_buffer *
new_buffer(struct nouveau_context *context,
           uint32_t width, uint32_t height,
           uint32_t format, uint32_t pitch)
{
	if (!buffer_layout_valid(width, height, format, pitch))
		return NULL;

	struct nouveau_buffer *buffer;

	if (!(buffer = malloc(sizeof *buffer)))
		return NULL;

	buffer_initialize(&buffer->base, &wld_buffer_impl,
	                  width, height, format, pitch);
	buffer->context = context;
	buffer->exporter.export = &export;
	wld_buffer_add_exporter(&buffer->base.base, &buffer->exporter);

	return buffer;
}

static inline uint32_t
roundup(uint32_t value, uint32_t alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}

struct buffer *
context_create_buffer(struct wld_context *base,
                      uint32_t width, uint32_t height,
                      uint32_t format, uint32_t flags)
{
	if (!buffer_dimensions_valid(width, height, format))
		return NULL;
	struct nouveau_context *context = nouveau_context(base);
	uint32_t bpp = format_bytes_per_pixel(format), storage_height = height;
	if (width > (UINT32_MAX - 63) / bpp || height > UINT32_MAX - 127)
		return NULL;
	uint32_t pitch = roundup(width * bpp, 64), bo_flags = NOUVEAU_BO_VRAM;
	union nouveau_bo_config config = {};
	if (flags & WLD_DRM_FLAG_SCANOUT)
		bo_flags |= NOUVEAU_BO_CONTIG;
	if (height > 0x40 && !(flags & WLD_FLAG_MAP)) {
		config.nvc0.tile_mode = 0x40;
		config.nvc0.memtype = 0xfe;
		storage_height = roundup(height, 0x80);
	} else {
		bo_flags |= NOUVEAU_BO_MAP;
	}
	if ((uint64_t)pitch * storage_height > UINT32_MAX)
		return NULL;
	struct nouveau_bo *bo = NULL;
	if (nouveau_bo_new(context->device, bo_flags, 0,
	                   (uint64_t)pitch * storage_height, &config, &bo) != 0)
		return NULL;
	struct nouveau_buffer *buffer = new_buffer(context, width, height, format, pitch);
	if (!buffer) {
		nouveau_bo_ref(NULL, &bo);
		return NULL;
	}
	buffer->bo = bo;
	return &buffer->base;
}

struct buffer *
context_import_buffer(struct wld_context *base,
                      uint32_t type, union wld_object object,
                      uint32_t width, uint32_t height,
                      uint32_t format, uint32_t pitch)
{
	if (!buffer_layout_valid(width, height, format, pitch))
		return NULL;

	struct nouveau_context *context = (void *)base;
	struct nouveau_buffer *buffer;
	struct nouveau_bo *bo = NULL;

	switch (type) {
	case WLD_DRM_OBJECT_PRIME_FD:
		if (nouveau_bo_prime_handle_ref(context->device,
		                                object.i, &bo)
		    != 0) {
			goto error0;
		}
		break;
	default:
		goto error0;
	}

	if (!(buffer = new_buffer(context, width, height, format, pitch)))
		goto error1;

	buffer->bo = bo;

	return &buffer->base;

error1:
	nouveau_bo_ref(NULL, &bo);
error0:
	return NULL;
}

void
context_destroy(struct wld_context *base)
{
	struct nouveau_context *context = nouveau_context(base);

	nouveau_client_del(&context->client);
	nouveau_device_del(&context->device);
	free(context);
}

/**** Renderer ****/
uint32_t
renderer_capabilities(struct wld_renderer *renderer,
                      struct buffer *buffer)
{
	if (buffer->base.impl == &wld_buffer_impl)
		return WLD_CAPABILITY_READ | WLD_CAPABILITY_WRITE;

	return 0;
}

bool
renderer_set_target(struct wld_renderer *base, struct buffer *buffer)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);

	if (buffer && buffer->base.impl != &wld_buffer_impl)
		return false;

	renderer->target = buffer ? nouveau_buffer(&buffer->base) : NULL;

	return true;
}

static inline void
nvc0_2d_use_buffer(struct nouveau_renderer *renderer,
                   struct nouveau_buffer *buffer,
                   uint16_t format_method, uint16_t format)
{
	uint32_t access = format == G80_2D_SRC_FORMAT ? NOUVEAU_BO_RD
	                                              : NOUVEAU_BO_WR;

	nvc0_2d_inline(renderer->pushbuf, format_method, format);

	if (buffer->bo->config.nvc0.memtype) {
		nvc0_2d(renderer->pushbuf, format_method + 0x04, 2,
		        0, buffer->bo->config.nvc0.tile_mode);
	} else {
		nvc0_2d_inline(renderer->pushbuf, format_method + 0x04, 1);
		nvc0_2d(renderer->pushbuf, format_method + 0x14, 1,
		        buffer->base.base.pitch);
	}

	nvc0_2d(renderer->pushbuf, format_method + 0x18, 4,
	        buffer->base.base.width, buffer->base.base.height,
	        buffer->bo->offset >> 32, buffer->bo->offset);
	nouveau_bufctx_refn(renderer->bufctx, 0, buffer->bo,
	                    NOUVEAU_BO_VRAM | access);
}

void
renderer_fill_rectangle(struct wld_renderer *base, uint32_t color,
                        int32_t x, int32_t y,
                        uint32_t width, uint32_t height)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);
	struct nouveau_buffer *dst = renderer->target;
	uint32_t format;

	if (!ensure_space(renderer->pushbuf, 18))
		return;

	format = nvc0_format(dst->base.base.format);

	nouveau_bufctx_reset(renderer->bufctx, 0);
	nvc0_2d_use_buffer(renderer, dst, G80_2D_DST_FORMAT, format);
	nvc0_2d(renderer->pushbuf, G80_2D_DRAW_SHAPE, 3,
	        G80_2D_DRAW_SHAPE_RECTANGLES, format, color);
	nouveau_pushbuf_bufctx(renderer->pushbuf, renderer->bufctx);

	if (nouveau_pushbuf_validate(renderer->pushbuf) != 0)
		return;

	nvc0_2d(renderer->pushbuf, G80_2D_DRAW_POINT32_X(0), 4,
	        x, y, x + width, y + height);
}

void
renderer_copy_rectangle(struct wld_renderer *base,
                        struct buffer *buffer_base,
                        int32_t dst_x, int32_t dst_y,
                        int32_t src_x, int32_t src_y,
                        uint32_t width, uint32_t height)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);

	if (buffer_base->base.impl != &wld_buffer_impl)
		return;

	struct nouveau_buffer *src = nouveau_buffer(&buffer_base->base),
	                      *dst = renderer->target;
	uint32_t src_format, dst_format;

	if (!ensure_space(renderer->pushbuf, 33))
		return;

	src_format = nvc0_format(src->base.base.format);
	dst_format = nvc0_format(dst->base.base.format);

	nouveau_bufctx_reset(renderer->bufctx, 0);
	nvc0_2d_use_buffer(renderer, src, G80_2D_SRC_FORMAT, src_format);
	nvc0_2d_use_buffer(renderer, dst, G80_2D_DST_FORMAT, dst_format);
	nouveau_pushbuf_bufctx(renderer->pushbuf, renderer->bufctx);

	if (nouveau_pushbuf_validate(renderer->pushbuf) != 0)
		return;

	nvc0_2d_inline(renderer->pushbuf, G80_GRAPH_SERIALIZE, 0);
	nvc0_2d_inline(renderer->pushbuf, G80_2D_BLIT_CONTROL,
	               G80_2D_BLIT_CONTROL_ORIGIN_CENTER
	                   | G80_2D_BLIT_CONTROL_FILTER_POINT_SAMPLE);
	nvc0_2d(renderer->pushbuf, G80_2D_BLIT_DST_X, 12,
	        dst_x, dst_y, width, height, 0, 1, 0, 1, 0, src_x, 0, src_y);

	renderer_flush(base);
}

void
renderer_draw_text(struct wld_renderer *base,
                   struct font *font, uint32_t color,
                   int32_t x, int32_t y, const char *text,
                   uint32_t length, struct wld_extents *extents)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);
	struct nouveau_buffer *dst = renderer->target;
	uint32_t format;
	int ret;
	struct glyph *glyph;
	FT_UInt glyph_index;
	uint32_t c, count;
	int32_t origin_x = x;

	if (length == UINT32_MAX)
		length = strlen(text);
	const char *cursor = text;
	uint32_t remaining = length;
	while ((ret = FcUtf8ToUcs4((FcChar8 *)cursor, &c, remaining)) > 0 && c) {
		cursor += ret;
		remaining -= ret;
		glyph_index = FT_Get_Char_Index(font->face, c);
		if (!font_ensure_glyph(font, glyph_index))
			continue;
		glyph = font->glyphs[glyph_index];
		uint64_t packed = glyph->bitmap.pixel_mode == FT_PIXEL_MODE_MONO && glyph->bitmap.pitch > 0
		                  ? (uint64_t)glyph->bitmap.pitch * glyph->bitmap.rows
		                  : ((uint64_t)glyph->bitmap.width + 7) / 8 * glyph->bitmap.rows;
		if (packed > 0x7ff * 4 ||
		    (glyph->bitmap.pixel_mode == FT_PIXEL_MODE_MONO && glyph->bitmap.pitch <= 0) ||
		    (glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY &&
		     glyph->bitmap.pixel_mode != FT_PIXEL_MODE_MONO)) {
			default_draw_text(base, renderer->context, font, color, x, y,
			                  text, length, extents);
			return;
		}
	}

	if (!ensure_space(renderer->pushbuf, 17))
		return;

	format = nvc0_format(dst->base.base.format);

	nouveau_bufctx_reset(renderer->bufctx, 0);
	nvc0_2d_use_buffer(renderer, dst, G80_2D_DST_FORMAT, format);
	nvc0_2d_inline(renderer->pushbuf, G80_2D_SIFC_BITMAP_ENABLE, 1);
	nvc0_2d(renderer->pushbuf, G80_2D_SIFC_BITMAP_FORMAT, 6,
	        G80_2D_SIFC_BITMAP_FORMAT_I1,
	        0, /* SIFC_FORMAT */
	        G80_2D_SIFC_BITMAP_LINE_PACK_MODE_ALIGN_BYTE,
	        0, color, /* SIFC_BITMAP_COLOR_BIT0, SIFC_BITMAP_COLOR_BIT1 */
	        0         /* SIFC_BITMAP_WRITE_BIT0_ENABLE */
	);
	nouveau_pushbuf_bufctx(renderer->pushbuf, renderer->bufctx);

	if (nouveau_pushbuf_validate(renderer->pushbuf) != 0)
		return;

	if (length == -1)
		length = strlen(text);

	while ((ret = FcUtf8ToUcs4((FcChar8 *)text, &c, length)) > 0 && c != '\0') {
		text += ret;
		length -= ret;
		glyph_index = FT_Get_Char_Index(font->face, c);

		if (!font_ensure_glyph(font, glyph_index))
			continue;

		glyph = font->glyphs[glyph_index];

		if (glyph->bitmap.width == 0 || glyph->bitmap.rows == 0)
			goto advance;

		if (glyph->bitmap.pixel_mode == FT_PIXEL_MODE_MONO) {
			if (glyph->bitmap.pitch <= 0 ||
			    (uint64_t)glyph->bitmap.pitch * glyph->bitmap.rows > 0x7ff * 4)
				goto advance;
			count = ((size_t)glyph->bitmap.pitch * glyph->bitmap.rows + 3) / 4;
		} else {
			uint32_t bytes_per_row = (glyph->bitmap.width + 7) / 8;
			if (glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY ||
			    (uint64_t)bytes_per_row * glyph->bitmap.rows > 0x7ff * 4)
				goto advance;
			count = ((size_t)bytes_per_row * glyph->bitmap.rows + 3) / 4;
		}

		if (!ensure_space(renderer->pushbuf, 12 + count))
			return;

		if (glyph->bitmap.pixel_mode == FT_PIXEL_MODE_MONO) {
			nvc0_2d(renderer->pushbuf, G80_2D_SIFC_WIDTH, 10,
			        /* Use the pitch instead of width to ensure the correct
				 * alignment is used. */
			        glyph->bitmap.pitch * 8, glyph->bitmap.rows,
			        0, 1, 0, 1,
			        0, origin_x + glyph->x, 0, y + glyph->y);
			nv_add_dword(renderer->pushbuf,
			             nvc0_command(GF100_COMMAND_TYPE_NON_INCREASING,
			                          GF100_SUBCHANNEL_2D,
			                          G80_2D_SIFC_DATA, count));
			nv_add_data(renderer->pushbuf, glyph->bitmap.buffer,
			            (size_t)glyph->bitmap.pitch * glyph->bitmap.rows);
		} else {
			uint32_t bytes_per_row = (glyph->bitmap.width + 7) / 8;
			uint32_t row;
			uint8_t *mono = malloc((size_t)bytes_per_row * glyph->bitmap.rows);
			uint8_t *dst = mono;

			if (!mono)
				return;

			for (row = 0; row < glyph->bitmap.rows; ++row) {
				const uint8_t *src = glyph->bitmap.buffer +
				                     ((ptrdiff_t)row * glyph->bitmap.pitch);
				pack_gray_row_to_mono(dst, src, glyph->bitmap.width);
				dst += bytes_per_row;
			}

			nvc0_2d(renderer->pushbuf, G80_2D_SIFC_WIDTH, 10,
			        bytes_per_row * 8, glyph->bitmap.rows,
			        0, 1, 0, 1,
			        0, origin_x + glyph->x, 0, y + glyph->y);
			nv_add_dword(renderer->pushbuf,
			             nvc0_command(GF100_COMMAND_TYPE_NON_INCREASING,
			                          GF100_SUBCHANNEL_2D,
			                          G80_2D_SIFC_DATA, count));
			nv_add_data(renderer->pushbuf, mono,
			            (size_t)bytes_per_row * glyph->bitmap.rows);
			free(mono);
		}

	advance:
		origin_x += glyph->advance;
	}

	if (extents)
		extents->advance = origin_x - x;
}

void
renderer_flush(struct wld_renderer *base)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);

	nouveau_pushbuf_kick(renderer->pushbuf, renderer->channel);
	nouveau_pushbuf_bufctx(renderer->pushbuf, NULL);
}

void
renderer_destroy(struct wld_renderer *base)
{
	struct nouveau_renderer *renderer = nouveau_renderer(base);

	nvc0_2d_finalize(renderer);
	nouveau_bufctx_del(&renderer->bufctx);
	nouveau_pushbuf_del(&renderer->pushbuf);
	nouveau_object_del(&renderer->channel);
	free(renderer);
}

/**** Buffer ****/
bool
buffer_map(struct buffer *base)
{
	struct nouveau_buffer *buffer = nouveau_buffer(&base->base);

	/* If the buffer is tiled, it cannot be mapped into virtual memory in order
	 * to appear linear like intel can do with map_gtt. */
	if (buffer->bo->config.nvc0.tile_mode)
		return false;

	if (nouveau_bo_map(buffer->bo, NOUVEAU_BO_RD | NOUVEAU_BO_WR,
	                   buffer->context->client)
	    != 0) {
		return false;
	}

	buffer->base.base.map = buffer->bo->map;

	return true;
}

bool
buffer_unmap(struct buffer *base)
{
	struct nouveau_buffer *buffer = nouveau_buffer(&base->base);

	if (munmap(buffer->bo->map, buffer->bo->size) == -1)
		return false;

	buffer->bo->map = NULL;
	base->base.map = NULL;

	return true;
}

void
buffer_destroy(struct buffer *base)
{
	struct nouveau_buffer *buffer = nouveau_buffer(&base->base);

	nouveau_bo_ref(NULL, &buffer->bo);
	free(buffer);
}
