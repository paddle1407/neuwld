/* wld: wld-private.h
 *
 * Copyright (c) 2013, 2014 Michael Forney
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

#ifndef WLD_PRIVATE_H
#define WLD_PRIVATE_H

#include "wld.h"

#include <assert.h>
#include <ft2build.h>
#include <stdbool.h>
#include <stdint.h>
#include FT_FREETYPE_H
#include FT_BITMAP_H

#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array)[0])
#if ENABLE_DEBUG
#define DEBUG(format, ...) \
	fprintf(stderr, "# %s: " format, __func__, ##__VA_ARGS__)
#else
#define DEBUG(format, ...)
#endif

#define EXPORT __attribute__((visibility("default")))
#define CONTAINER_OF(ptr, type, member) \
	((type *)((uintptr_t)ptr - offsetof(type, member)))
#define IMPL(impl_type, base_type)                                          \
	static inline struct impl_type *impl_type(struct base_type *object) \
	{                                                                   \
		assert(object->impl == &base_type##_impl);                  \
		return (struct impl_type *)object;                          \
	}

struct wld_font_context {
	FT_Library library;
};

struct glyph {
	FT_Bitmap bitmap;
	uint64_t serial; /* Identifies this allocation across font reloads. */

	/**
	 * The offset from the origin to the top left corner of the bitmap.
	 */
	int16_t x, y;

	/**
	 * The width to advance to the origin of the next character.
	 */
	uint16_t advance;
};

struct font {
	struct wld_font base;

	struct wld_font_context *context;
	FT_Face face;
	struct glyph **glyphs;
};

struct wld_context_impl {
	struct wld_renderer *(*create_renderer)(struct wld_context *context);
	struct buffer *(*create_buffer)(struct wld_context *context,
	                                uint32_t width, uint32_t height,
	                                uint32_t format, uint32_t flags);
	struct buffer *(*import_buffer)(struct wld_context *context,
	                                uint32_t type, union wld_object object,
	                                uint32_t width, uint32_t height,
	                                uint32_t format, uint32_t pitch);
	struct wld_surface *(*create_surface)(struct wld_context *context,
	                                      uint32_t width, uint32_t height,
	                                      uint32_t format, uint32_t flags);
	/**
	 * Optional. Fills 'modifiers' with the DRM format modifiers this context
	 * can import for 'format', returning the count, or -1 if unknown.
	 */
	int (*query_modifiers)(struct wld_context *context, uint32_t format,
	                       uint64_t *modifiers, int max);
	void (*destroy)(struct wld_context *context);
};

struct wld_renderer_impl {
	uint32_t (*capabilities)(struct wld_renderer *renderer,
	                         struct buffer *buffer);
	bool (*set_target)(struct wld_renderer *renderer, struct buffer *buffer);
	/**
	 * Optional. Confines drawing to a box of the target until the target
	 * changes; NULL lifts it.
	 */
	void (*set_clip)(struct wld_renderer *renderer, const pixman_box32_t *box);
	void (*fill_rectangle)(struct wld_renderer *renderer,
	                       uint32_t color, int32_t x, int32_t y,
	                       uint32_t width, uint32_t height);
	void (*fill_region)(struct wld_renderer *renderer,
	                    uint32_t color, pixman_region32_t *region);
	void (*copy_rectangle)(struct wld_renderer *renderer, struct buffer *src,
	                       int32_t dst_x, int32_t dst_y,
	                       int32_t src_x, int32_t src_y,
	                       uint32_t width, uint32_t height);
	void (*copy_region)(struct wld_renderer *renderer, struct buffer *src,
	                    int32_t dst_x, int32_t dst_y,
	                    pixman_region32_t *region);
	/**
	 * Optional. When NULL, wld_blend_region() falls back to mapping both
	 * buffers and blending on the CPU, which on an accelerated backend costs
	 * a full readback and pipeline stall per call.
	 */
	void (*blend_region)(struct wld_renderer *renderer, struct buffer *src,
	                     int32_t dst_x, int32_t dst_y,
	                     pixman_region32_t *region);
	/**
	 * Optional. Blends a source rectangle into a destination rectangle of a
	 * different size. When NULL, wld_blend_scaled() falls back to mapping
	 * both buffers and scaling on the CPU, which on an accelerated backend
	 * costs a full readback and pipeline stall per call.
	 */
	void (*blend_scaled)(struct wld_renderer *renderer, struct buffer *src,
	                     const struct wld_rect *dst,
	                     const struct wld_frect *src_rect);
	void (*draw_circle)(struct wld_renderer *renderer, uint32_t color,
				int32_t x, int32_t y, uint32_t r, bool fill);
	void (*draw_line)(struct wld_renderer *renderer, uint32_t color,
			 int32_t x1, int32_t y1, int32_t x2, int32_t y2);
	void (*draw_text)(struct wld_renderer *renderer,
	                  struct font *font, uint32_t color,
	                  int32_t x, int32_t y, const char *text, uint32_t length,
	                  struct wld_extents *extents);
	void (*flush)(struct wld_renderer *renderer);
	/**
	 * Optional. Reads back the current target into 'data'. Lets a caller
	 * capture what an accelerated backend composited, which it cannot do by
	 * mapping the target: buffers imported from clients are not CPU
	 * accessible.
	 */
	bool (*read_pixels)(struct wld_renderer *renderer, int32_t x, int32_t y,
	                    uint32_t width, uint32_t height, uint32_t pitch,
	                    void *data);
	/**
	 * Optional. Orders subsequent rendering after a DRM sync_file fence,
	 * without consuming the caller's file descriptor, and without blocking.
	 * A negative fence_fd only probes support. NULL when the backend has no
	 * notion of fences.
	 */
	bool (*wait_fence)(struct wld_renderer *renderer, int fence_fd);
	/**
	 * Optional. Submits all rendering so far and returns a sync_file that
	 * signals when it completes, or -1 once it has completed instead.
	 */
	int (*export_fence)(struct wld_renderer *renderer);
	void (*destroy)(struct wld_renderer *renderer);
};

struct buffer {
	struct wld_buffer base;

	unsigned ref, map_ref;
	struct wld_exporter *exporters;
	struct wld_destructor *destructors;
};

struct wld_buffer_impl {
	bool (*map)(struct buffer *buffer);
	bool (*unmap)(struct buffer *buffer);
	void (*flush)(struct buffer *buffer);
	/**
	 * Optional. Told, ahead of flush(), which part of the buffer a CPU
	 * renderer wrote since it became the target, so a backend that mirrors
	 * the pixels elsewhere can refresh only that part. NULL means the writes
	 * could have gone anywhere. A flush() with no damage() before it must
	 * still assume the whole buffer changed.
	 */
	void (*damage)(struct buffer *buffer, pixman_region32_t *region);
	/**
	 * Optional. Copies a region of the caller's pixels into the buffer
	 * without going through a mapping. See wld_buffer_upload().
	 */
	bool (*upload)(struct buffer *buffer, const void *pixels, uint32_t pitch,
	               pixman_region32_t *region);
	void (*destroy)(struct buffer *buffer);
};

struct wld_surface_impl {
	pixman_region32_t *(*damage)(struct wld_surface *surface,
	                             pixman_region32_t *damage);
	struct buffer *(*back)(struct wld_surface *surface);
	struct buffer *(*take)(struct wld_surface *surface);
	bool (*release)(struct wld_surface *surface, struct buffer *buffer);
	bool (*swap)(struct wld_surface *surface);
	void (*destroy)(struct wld_surface *surface);
};

struct buffer_socket {
	const struct buffer_socket_impl *impl;
};

struct buffer_socket_impl {
	bool (*attach)(struct buffer_socket *socket, struct buffer *buffer);
	void (*process)(struct buffer_socket *socket);
	void (*destroy)(struct buffer_socket *socket);
};

bool font_ensure_glyph(struct font *font, FT_UInt glyph_index);

/**
 * Returns the number of bytes per pixel for the given format.
 */
static inline uint8_t
format_bytes_per_pixel(enum wld_format format)
{
	switch (format) {
	case WLD_FORMAT_ARGB8888:
	case WLD_FORMAT_XRGB8888:
		return 4;
	default:
		return 0;
	}
}
/*
 * The pixman transform that maps the destination rectangle `dst` back onto
 * the source rectangle `src`, which is the direction pixman samples in.
 *
 * The half-pixel terms are the difference between an edge coordinate and a
 * centre one. pixman hands the filter the coordinate of the destination
 * pixel's centre and expects back the centre of the source pixel to sample,
 * while `src` names the rectangle's edges; without the correction the image
 * lands half a source pixel off, which at large minification is most of a
 * destination pixel.
 *
 * At scale 1 with an integral origin it reduces to a plain translation, so a
 * scaled blit that is not scaling anything still copies exactly.
 */
static inline void
scaled_transform(pixman_transform_t *transform,
                 const struct wld_rect *dst, const struct wld_frect *src)
{
	double scale_x = src->width / (double)dst->width;
	double scale_y = src->height / (double)dst->height;

	pixman_transform_init_identity(transform);
	transform->matrix[0][0] = pixman_double_to_fixed(scale_x);
	transform->matrix[1][1] = pixman_double_to_fixed(scale_y);
	transform->matrix[0][2] = pixman_double_to_fixed(src->x + scale_x / 2 - 0.5);
	transform->matrix[1][2] = pixman_double_to_fixed(src->y + scale_y / 2 - 0.5);
}


static inline pixman_format_code_t
format_wld_to_pixman(uint32_t format)
{
	switch (format) {
	case WLD_FORMAT_ARGB8888:
		return PIXMAN_a8r8g8b8;
	case WLD_FORMAT_XRGB8888:
		return PIXMAN_x8r8g8b8;
	default:
		return 0;
	}
}

static inline uint32_t
format_pixman_to_wld(pixman_format_code_t format)
{
	switch (format) {
	case PIXMAN_a8r8g8b8:
		return WLD_FORMAT_ARGB8888;
	case PIXMAN_x8r8g8b8:
		return WLD_FORMAT_XRGB8888;
	default:
		return 0;
	}
}

/**
 * This default fill_region method is implemented in terms of fill_rectangle.
 */
void default_fill_region(struct wld_renderer *renderer, uint32_t color,
                         pixman_region32_t *region);

/**
 * This default copy_region method is implemented in terms of copy_rectangle.
 */
void default_copy_region(struct wld_renderer *renderer, struct buffer *buffer,
                         int32_t dst_x, int32_t dst_y,
                         pixman_region32_t *region);

struct wld_surface *default_create_surface(struct wld_context *context,
                                           uint32_t width, uint32_t height,
                                           uint32_t format, uint32_t flags);

struct wld_surface *buffered_surface_create(struct wld_context *context,
                                            uint32_t width, uint32_t height,
                                            uint32_t format, uint32_t flags,
                                            struct buffer_socket *socket);

void context_initialize(struct wld_context *context,
                        const struct wld_context_impl *impl);

void renderer_initialize(struct wld_renderer *renderer,
                         const struct wld_renderer_impl *impl);

void buffer_initialize(struct buffer *buffer,
                       const struct wld_buffer_impl *impl,
                       uint32_t width, uint32_t height,
                       uint32_t format, uint32_t pitch);

void surface_initialize(struct wld_surface *surface,
                        const struct wld_surface_impl *impl);

#endif
