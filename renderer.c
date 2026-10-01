/* wld: renderer.c
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

#include "wld-private.h"

#include <math.h>
#include "pixman.h"

void
default_fill_region(struct wld_renderer *renderer, uint32_t color, pixman_region32_t *region)
{
	pixman_box32_t *box;
	int num_boxes;

	box = pixman_region32_rectangles(region, &num_boxes);

	while (num_boxes--) {
		wld_fill_rectangle(renderer, color, box->x1, box->y1,
		                   box->x2 - box->x1, box->y2 - box->y1);
		++box;
	}
}

void
default_copy_region(struct wld_renderer *renderer, struct buffer *buffer,
                    int32_t dst_x, int32_t dst_y,
                    pixman_region32_t *region)
{
	pixman_box32_t *box;
	int num_boxes;

	box = pixman_region32_rectangles(region, &num_boxes);

	while (num_boxes--) {
		int64_t x = (int64_t)dst_x + box->x1;
		int64_t y = (int64_t)dst_y + box->y1;
		if (x < INT32_MIN || x > INT32_MAX || y < INT32_MIN || y > INT32_MAX) {
			++box;
			continue;
		}
		wld_copy_rectangle(renderer, &buffer->base,
		                   x, y,
		                   box->x1, box->y1,
		                   box->x2 - box->x1, box->y2 - box->y1);
		++box;
	}
}

void
renderer_initialize(struct wld_renderer *renderer, const struct wld_renderer_impl *impl)
{
	*((const struct wld_renderer_impl **)&renderer->impl) = impl;
	renderer->target = NULL;
	renderer->clip_enabled = false;
}

EXPORT
void
wld_destroy_renderer(struct wld_renderer *renderer)
{
	wld_flush(renderer);
	renderer->impl->destroy(renderer);
}

EXPORT
uint32_t
wld_capabilities(struct wld_renderer *renderer, struct wld_buffer *buffer)
{
	return buffer ? renderer->impl->capabilities(renderer, (struct buffer *)buffer) : 0;
}

EXPORT
bool
wld_wait_fence(struct wld_renderer *renderer, int fence_fd)
{
	if (!renderer->impl->wait_fence)
		return false;

	return renderer->impl->wait_fence(renderer, fence_fd);
}

EXPORT
int
wld_export_fence(struct wld_renderer *renderer)
{
	/* Without a fence of its own, a backend's flush is already a barrier. */
	if (!renderer->impl->export_fence)
		return -1;

	return renderer->impl->export_fence(renderer);
}

EXPORT
bool
wld_set_target_buffer(struct wld_renderer *renderer, struct wld_buffer *buffer)
{
	struct wld_buffer *old_target = renderer->target;
	if (buffer)
		wld_buffer_reference(buffer);
	if (!renderer->impl->set_target(renderer, (struct buffer *)buffer)) {
		if (buffer)
			wld_buffer_unreference(buffer);
		return false;
	}

	renderer->target = buffer;
	if (old_target)
		wld_buffer_unreference(old_target);
	renderer->clip_enabled = false;

	return true;
}

EXPORT
void
wld_set_clip(struct wld_renderer *renderer, const pixman_box32_t *box)
{
	renderer->clip_enabled = box != NULL;
	if (box) {
		renderer->clip = *box;
		if (box->x2 < box->x1) renderer->clip.x2 = box->x1;
		if (box->y2 < box->y1) renderer->clip.y2 = box->y1;
	}
	if (renderer->impl->set_clip)
		renderer->impl->set_clip(renderer, box ? &renderer->clip : NULL);
}

EXPORT
bool
wld_set_target_surface(struct wld_renderer *renderer, struct wld_surface *surface)
{
	struct buffer *back_buffer;

	if (!(back_buffer = surface->impl->back(surface)))
		return false;

	return wld_set_target_buffer(renderer, &back_buffer->base);
}

/* Clip before narrowing backend coordinates or submitting hardware commands. */
static bool
clip_rectangle(struct wld_renderer *renderer, int32_t *x, int32_t *y,
               uint32_t *width, uint32_t *height)
{
	if (!renderer->target || !*width || !*height)
		return false;
	int64_t x1 = *x, y1 = *y, x2 = x1 + *width, y2 = y1 + *height;
	if (x1 < 0) x1 = 0;
	if (y1 < 0) y1 = 0;
	if (x2 > renderer->target->width) x2 = renderer->target->width;
	if (y2 > renderer->target->height) y2 = renderer->target->height;
	if (renderer->clip_enabled) {
		if (x1 < renderer->clip.x1) x1 = renderer->clip.x1;
		if (y1 < renderer->clip.y1) y1 = renderer->clip.y1;
		if (x2 > renderer->clip.x2) x2 = renderer->clip.x2;
		if (y2 > renderer->clip.y2) y2 = renderer->clip.y2;
	}
	if (x2 <= x1 || y2 <= y1)
		return false;
	*x = x1; *y = y1; *width = x2 - x1; *height = y2 - y1;
	return true;
}

EXPORT
void
wld_fill_rectangle(struct wld_renderer *renderer, uint32_t color,
                   int32_t x, int32_t y, uint32_t width, uint32_t height)
{
	if (clip_rectangle(renderer, &x, &y, &width, &height))
		renderer->impl->fill_rectangle(renderer, color, x, y, width, height);
}

EXPORT
void
wld_fill_region(struct wld_renderer *renderer, uint32_t color, pixman_region32_t *region)
{
	if (renderer->target && region)
		renderer->impl->fill_region(renderer, color, region);
}

EXPORT
void
wld_copy_rectangle(struct wld_renderer *renderer,
                   struct wld_buffer *buffer,
                   int32_t dst_x, int32_t dst_y,
                   int32_t src_x, int32_t src_y,
                   uint32_t width, uint32_t height)
{
	int32_t old_x = dst_x, old_y = dst_y;
	if (!buffer || !clip_rectangle(renderer, &dst_x, &dst_y, &width, &height))
		return;
	int64_t sx = (int64_t)src_x + dst_x - old_x;
	int64_t sy = (int64_t)src_y + dst_y - old_y;
	if (sx < 0) {
		if (-sx >= width) return;
		dst_x -= sx; width += sx; sx = 0;
	}
	if (sy < 0) {
		if (-sy >= height) return;
		dst_y -= sy; height += sy; sy = 0;
	}
	if (sx >= buffer->width || sy >= buffer->height)
		return;
	if (width > buffer->width - sx) width = buffer->width - sx;
	if (height > buffer->height - sy) height = buffer->height - sy;
	renderer->impl->copy_rectangle(renderer, (struct buffer *)buffer,
	                               dst_x, dst_y, sx, sy, width, height);
}

EXPORT
void
wld_copy_region(struct wld_renderer *renderer,
                struct wld_buffer *buffer,
                int32_t dst_x, int32_t dst_y, pixman_region32_t *region)
{
	if (!renderer->target || !buffer || !region)
		return;
	renderer->impl->copy_region(renderer, (struct buffer *)buffer,
	                            dst_x, dst_y, region);
}

EXPORT
void
wld_blend_region(struct wld_renderer *renderer, struct wld_buffer *buffer,
                 int32_t dst_x, int32_t dst_y, pixman_region32_t *region)
{
	pixman_image_t *src = NULL, *dst = NULL;
	pixman_region32_t clip;
	pixman_box32_t *extents;

	if (!renderer->target || !buffer || !region || !pixman_region32_not_empty(region))
		return;

	/* An accelerated backend blends on the GPU and skips the readback below. */
	if (renderer->impl->blend_region) {
		renderer->impl->blend_region(renderer, (struct buffer *)buffer,
		                             dst_x, dst_y, region);
		return;
	}

	/* Complete accelerator writes before accessing the buffers on the CPU. */
	renderer->impl->flush(renderer);
	if (!wld_map(buffer))
		return;
	if (!wld_map(renderer->target))
		goto unmap_src;

	src = pixman_image_create_bits(format_wld_to_pixman(buffer->format),
	                               buffer->width, buffer->height,
	                               buffer->map, buffer->pitch);
	dst = pixman_image_create_bits(format_wld_to_pixman(renderer->target->format),
	                               renderer->target->width,
	                               renderer->target->height,
	                               renderer->target->map,
	                               renderer->target->pitch);
	if (!src || !dst)
		goto destroy;

	pixman_region32_init(&clip);
	pixman_region32_copy(&clip, region);
	pixman_region32_translate(&clip, dst_x, dst_y);
	pixman_region32_intersect_rect(&clip, &clip, 0, 0,
	                               renderer->target->width, renderer->target->height);
	if (renderer->clip_enabled) {
		pixman_region32_t confinement;
		pixman_region32_init_with_extents(&confinement, &renderer->clip);
		pixman_region32_intersect(&clip, &clip, &confinement);
		pixman_region32_fini(&confinement);
	}
	extents = pixman_region32_extents(region);
	pixman_image_set_clip_region32(dst, &clip);
	pixman_image_composite32(PIXMAN_OP_OVER, src, NULL, dst,
	                         extents->x1, extents->y1, 0, 0,
	                         extents->x1 + dst_x, extents->y1 + dst_y,
	                         extents->x2 - extents->x1,
	                         extents->y2 - extents->y1);
	/* The renderer did not see this write, so report it on its behalf. */
	if (((struct buffer *)renderer->target)->base.impl->damage) {
		((struct buffer *)renderer->target)->base.impl->damage(
		    (struct buffer *)renderer->target, &clip);
	}
	pixman_region32_fini(&clip);

destroy:
	if (src)
		pixman_image_unref(src);
	if (dst)
		pixman_image_unref(dst);
	wld_unmap(renderer->target);
unmap_src:
	wld_unmap(buffer);
}

EXPORT
void
wld_blend_scaled(struct wld_renderer *renderer, struct wld_buffer *buffer,
                 const struct wld_rect *dst, const struct wld_frect *src)
{
	pixman_image_t *source = NULL, *target = NULL;
	pixman_transform_t transform;
	pixman_region32_t damage;

	if (!renderer->target || !buffer || !dst || !src || dst->width == 0 || dst->height == 0)
		return;
	if (!isfinite(src->x) || !isfinite(src->y) || !isfinite(src->width) ||
	    !isfinite(src->height) || src->width <= 0.0 || src->height <= 0.0)
		return;

	/* An accelerated backend scales on the GPU and skips the readback below. */
	if (renderer->impl->blend_scaled) {
		renderer->impl->blend_scaled(renderer, (struct buffer *)buffer, dst,
		                             src);
		return;
	}

	/* Complete accelerator writes before accessing the buffers on the CPU. */
	renderer->impl->flush(renderer);
	if (!wld_map(buffer))
		return;
	if (!wld_map(renderer->target))
		goto unmap_source;

	source = pixman_image_create_bits(format_wld_to_pixman(buffer->format),
	                                  buffer->width, buffer->height,
	                                  buffer->map, buffer->pitch);
	target = pixman_image_create_bits(
	    format_wld_to_pixman(renderer->target->format),
	    renderer->target->width, renderer->target->height,
	    renderer->target->map, renderer->target->pitch);
	if (!source || !target)
		goto destroy_images;

	if (!scaled_transform(&transform, dst, src))
		goto destroy_images;
	pixman_region32_init_rect(&damage, dst->x, dst->y, dst->width, dst->height);
	pixman_region32_intersect_rect(&damage, &damage, 0, 0,
	                               renderer->target->width, renderer->target->height);
	if (renderer->clip_enabled) {
		pixman_region32_t confinement;
		pixman_region32_init_with_extents(&confinement, &renderer->clip);
		pixman_region32_intersect(&damage, &damage, &confinement);
		pixman_region32_fini(&confinement);
	}
	pixman_image_set_clip_region32(target, &damage);
	pixman_image_set_transform(source, &transform);
	pixman_image_set_filter(source, PIXMAN_FILTER_BILINEAR, NULL, 0);
	/*
	 * The transform carries the source origin, so the composite reads from
	 * (0, 0) and every destination pixel is asked for by its own coordinate.
	 */
	pixman_image_composite32(PIXMAN_OP_OVER, source, NULL, target,
	                         0, 0, 0, 0, dst->x, dst->y,
	                         dst->width, dst->height);
	if (renderer->target->impl->damage)
		renderer->target->impl->damage((struct buffer *)renderer->target, &damage);
	pixman_region32_fini(&damage);

destroy_images:
	if (source)
		pixman_image_unref(source);
	if (target)
		pixman_image_unref(target);
	wld_unmap(renderer->target);
unmap_source:
	wld_unmap(buffer);
}


/* https://en.wikipedia.org/wiki/Midpoint_circle_algorithm */
static void
circle_points(struct wld_renderer *renderer, uint32_t color,
			 int32_t x1, int32_t y1, int32_t x2, int32_t y2, bool fill)
{
	/* hacky */
	if (fill) {
		wld_fill_rectangle(renderer, color, x1-x2, y1+y2, 2*x2 + 1, 1);
		wld_fill_rectangle(renderer, color, x1-x2, y1-y2, 2*x2 + 1, 1);
		wld_fill_rectangle(renderer, color, x1-y2, y1+x2, 2*y2 + 1, 1);
		wld_fill_rectangle(renderer, color, x1-y2, y1-x2, 2*y2 + 1, 1);
	}
	
	else {
		wld_fill_rectangle(renderer, color, x1+x2, y1+y2, 1, 1);
		wld_fill_rectangle(renderer, color, x1-x2, y1+y2, 1, 1);
		wld_fill_rectangle(renderer, color, x1+x2, y1-y2, 1, 1);
		wld_fill_rectangle(renderer, color, x1-x2, y1-y2, 1, 1);
		wld_fill_rectangle(renderer, color, x1+y2, y1+x2, 1, 1);
		wld_fill_rectangle(renderer, color, x1-y2, y1+x2, 1, 1);
		wld_fill_rectangle(renderer, color, x1+y2, y1-x2, 1, 1);
		wld_fill_rectangle(renderer, color, x1-y2, y1-x2, 1, 1);
	}
}

EXPORT
void
wld_draw_circle(struct wld_renderer *renderer, uint32_t color, 
				int32_t x, int32_t y, uint32_t r, bool fill)
{
	if (r > INT32_MAX / 8 || (int64_t)x - r < INT32_MIN ||
	    (int64_t)x + r > INT32_MAX || (int64_t)y - r < INT32_MIN ||
	    (int64_t)y + r > INT32_MAX)
		return;
	int32_t x1 = 0, y1 = r;
	int32_t d = 3 - 2 * r;
	circle_points(renderer, color, x, y, x1, y1, fill);

	while (y1 >= x1) {
		if (d > 0) {
			y1--;
			d = d + 4 * (x1 - y1) + 10;
		}
		else
			d = d + 4 * x1 + 6;

		x1++;

		circle_points(renderer, color, x, y, x1, y1, fill);
	}
}

EXPORT
void
wld_draw_line(struct wld_renderer *renderer, uint32_t color,
			 int32_t x1, int32_t y1, int32_t x2, int32_t y2)
{
	int64_t dx = llabs((int64_t)x2 - x1), dy = -llabs((int64_t)y2 - y1);
	int32_t sx = x1 < x2 ? 1 : -1, sy = y1 < y2 ? 1 : -1;
	int64_t err = dx + dy, e2;

	while(true) {
		wld_fill_rectangle(renderer, color, x1, y1, 1, 1);

		if (x1==x2 && y1==y2)
			break;
		
		e2 = 2*err;
		
		if (e2 >= dy) {
			err += dy;
			x1 += sx;
		}

		if (e2 <= dx) {
			err += dx;
			y1 += sy;
		}
	}
}

void
default_draw_text(struct wld_renderer *renderer, struct wld_context *context,
                  struct font *font, uint32_t color, int32_t x, int32_t y,
                  const char *text, uint32_t length, struct wld_extents *extents)
{
	struct wld_buffer *target = renderer->target, *staging = NULL;
	struct wld_renderer *copy = NULL;
	if (!target)
		return;
	/* Submit the target's earlier writes before mapping it or copying them. */
	renderer->impl->flush(renderer);
	struct wld_renderer *cpu = wld_create_renderer(wld_pixman_context);
	if (!cpu)
		return;
	if (!wld_set_target_buffer(cpu, target)) {
		/* A tiled target cannot be mapped linearly. Preserve its background
		 * in a mapped buffer belonging to the same hardware context, then
		 * blend on the CPU and copy the resulting pixels back, without
		 * introducing a cross-backend source or applying alpha twice. */
		if (!context)
			goto done;
		staging = wld_create_buffer(context, target->width, target->height,
		                            target->format, WLD_FLAG_MAP);
		copy = wld_create_renderer(context);
		if (!staging || !copy ||
		    !(wld_capabilities(copy, target) & WLD_CAPABILITY_READ) ||
		    !(wld_capabilities(copy, staging) & WLD_CAPABILITY_WRITE) ||
		    !(wld_capabilities(renderer, staging) & WLD_CAPABILITY_READ) ||
		    !wld_set_target_buffer(copy, staging))
			goto done;
		wld_copy_rectangle(copy, target, 0, 0, 0, 0,
		                   target->width, target->height);
		wld_flush(copy);
		if (!wld_set_target_buffer(cpu, staging))
			goto done;
	}
	if (renderer->clip_enabled)
		wld_set_clip(cpu, &renderer->clip);
	wld_draw_text(cpu, &font->base, color, x, y, text, length, extents);
	wld_flush(cpu);
	if (staging) {
		/* The rectangle wrapper retains the original target confinement. */
		wld_copy_rectangle(renderer, staging, 0, 0, 0, 0,
		                   target->width, target->height);
		/* Kernel/driver references must cover staging until this submission. */
		renderer->impl->flush(renderer);
	}
done:
	wld_destroy_renderer(cpu);
	if (copy)
		wld_destroy_renderer(copy);
	if (staging)
		wld_buffer_unreference(staging);
}

EXPORT
void
wld_draw_text(struct wld_renderer *renderer,
              struct wld_font *font_base, uint32_t color,
              int32_t x, int32_t y, const char *text, uint32_t length,
              struct wld_extents *extents)
{
	struct font *font = (void *)font_base;
	if (extents)
		extents->advance = 0;
	if (!renderer->target || !font || !text)
		return;

	renderer->impl->draw_text(renderer, font, color, x, y, text, length,
	                          extents);
}

EXPORT
bool
wld_read_pixels(struct wld_renderer *renderer, int32_t x, int32_t y,
                uint32_t width, uint32_t height, uint32_t pitch, void *data)
{
	if (!renderer->impl->read_pixels || !renderer->target)
		return false;

	return renderer->impl->read_pixels(renderer, x, y, width, height, pitch,
	                                   data);
}

EXPORT
void
wld_flush(struct wld_renderer *renderer)
{
	renderer->impl->flush(renderer);
	if (renderer->target && ((struct buffer *)renderer->target)->base.impl->flush)
		((struct buffer *)renderer->target)->base.impl->flush(
		    (struct buffer *)renderer->target);
	wld_set_target_buffer(renderer, NULL);
}
