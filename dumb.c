/* wld: dumb.c
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

#include "drm-private.h"
#include "drm.h"
#include "pixman.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>

#include <errno.h>
#include <string.h>

struct dumb_handle {
	struct dumb_handle *next;
	uint32_t handle;
	unsigned references;
	bool allocated;
};

struct dumb_context {
	struct wld_context base;
	int fd;
	struct dumb_handle *handles;
};

struct dumb_buffer {
	struct buffer base;
	struct wld_exporter exporter;
	struct dumb_context *context;
	uint32_t handle;
	struct dumb_handle *handle_ref;
	void *scanout;
	bool shadowed;
};

#define BUFFER_IMPLEMENTS_FLUSH
#include "interface/buffer.h"
#include "interface/context.h"
#define DRM_DRIVER_NAME dumb
#include "interface/drm.h"
IMPL(dumb_context, wld_context)
IMPL(dumb_buffer, wld_buffer)

const struct wld_context_impl *dumb_context_impl = &wld_context_impl;

bool
driver_device_supported(uint32_t vendor_id, uint32_t device_id)
{
	return true;
}

struct wld_context *
driver_create_context(int drm_fd)
{
	struct dumb_context *context;

	if (!(context = malloc(sizeof *context)))
		return NULL;

	context_initialize(&context->base, &wld_context_impl);
	context->fd = drm_fd;
	context->handles = NULL;

	return &context->base;
}

struct wld_renderer *
context_create_renderer(struct wld_context *context)
{
	return wld_create_renderer(wld_pixman_context);
}

static bool export(struct wld_exporter *exporter, struct wld_buffer *base,
                   uint32_t type, union wld_object *object)
{
	struct dumb_buffer *buffer = dumb_buffer(base);

	switch (type) {
	case WLD_DRM_OBJECT_HANDLE:
		object->u32 = buffer->handle;
		return true;
	case WLD_DRM_OBJECT_PRIME_FD:
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

static struct buffer *
new_buffer(struct dumb_context *context,
           uint32_t width, uint32_t height,
           uint32_t format, uint32_t handle,
           unsigned long pitch, bool shadowed, bool allocated)
{
	if (!buffer_layout_valid(width, height, format, pitch))
		return NULL;

	struct dumb_buffer *buffer;

	if (!(buffer = malloc(sizeof *buffer)))
		return NULL;

	struct dumb_handle *reference;
	for (reference = context->handles; reference; reference = reference->next)
		if (reference->handle == handle)
			break;
	if (!reference) {
		if (!(reference = malloc(sizeof *reference))) {
			free(buffer);
			return NULL;
		}
		*reference = (struct dumb_handle){.next = context->handles,
		                                .handle = handle, .allocated = allocated};
		context->handles = reference;
	}
	++reference->references;
	buffer->handle_ref = reference;
	buffer_initialize(&buffer->base, &wld_buffer_impl,
	                  width, height, format, pitch);
	buffer->context = context;
	buffer->handle = handle;
	buffer->scanout = NULL;
	buffer->shadowed = shadowed;
	buffer->exporter.export = &export;
	wld_buffer_add_exporter(&buffer->base.base, &buffer->exporter);

	return &buffer->base;
}

struct buffer *
context_create_buffer(struct wld_context *base,
                      uint32_t width, uint32_t height,
                      uint32_t format, uint32_t flags)
{
	if (!buffer_dimensions_valid(width, height, format))
		return NULL;

	struct dumb_context *context = dumb_context(base);
	struct buffer *buffer;
	struct drm_mode_create_dumb create_dumb = {
		.height = height,
		.width = width,
		.bpp = format_bytes_per_pixel(format) * 8,
	};

	if (drmIoctl(context->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_dumb) != 0)
		goto error0;

	buffer = new_buffer(context, width, height, format,
	                    create_dumb.handle, create_dumb.pitch,
	                    flags & WLD_DRM_FLAG_SCANOUT, true);

	if (!buffer)
		goto error1;

	return buffer;

error1 : {
	struct drm_mode_destroy_dumb destroy_dumb = {
		.handle = create_dumb.handle
	};

	drmIoctl(context->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_dumb);
}
error0:
	return NULL;
}

struct buffer *
context_import_buffer(struct wld_context *base,
                      uint32_t type, union wld_object object,
                      uint32_t width, uint32_t height,
                      uint32_t format, uint32_t pitch)
{
	if (!buffer_layout_valid(width, height, format, pitch))
		return NULL;

	struct dumb_context *context = dumb_context(base);
	uint32_t handle;

	switch (type) {
	case WLD_DRM_OBJECT_PRIME_FD:
		if (drmPrimeFDToHandle(context->fd, object.i, &handle) != 0)
			return NULL;
		break;
	default:
		return NULL;
	}

	struct buffer *buffer = new_buffer(context, width, height, format, handle, pitch, false, false);
	if (!buffer) {
		/* An alias already owned by another wrapper is not ours to close. */
		struct dumb_handle *reference;
		for (reference = context->handles; reference; reference = reference->next)
			if (reference->handle == handle)
				break;
		if (!reference) {
			struct drm_gem_close close_arg = { .handle = handle };
			drmIoctl(context->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
		}
	}
	return buffer;
}

void
context_destroy(struct wld_context *base)
{
	struct dumb_context *context = dumb_context(base);

	/* The fd belongs to the caller of wld_drm_create_context. */
	free(context);
}

/**** Buffer ****/

bool
buffer_map(struct buffer *base)
{
	struct dumb_buffer *buffer = dumb_buffer(&base->base);
	struct drm_mode_map_dumb map_dumb = { .handle = buffer->handle };
	void *data;

	if (drmIoctl(buffer->context->fd, DRM_IOCTL_MODE_MAP_DUMB,
	             &map_dumb)
	    != 0) {
		return false;
	}

	data = mmap(NULL, (size_t)buffer->base.base.pitch * buffer->base.base.height,
	            PROT_READ | PROT_WRITE, MAP_SHARED,
	            buffer->context->fd, map_dumb.offset);

	if (data == MAP_FAILED)
		return false;

	buffer->scanout = data;
	if (buffer->shadowed) {
		buffer->base.base.map = calloc(buffer->base.base.height,
		                               buffer->base.base.pitch);
		if (!buffer->base.base.map) {
			munmap(data, (size_t)buffer->base.base.pitch * buffer->base.base.height);
			buffer->scanout = NULL;
			return false;
		}
	} else {
		buffer->base.base.map = data;
	}

	return true;
}

void
buffer_flush(struct buffer *base)
{
	struct dumb_buffer *buffer = dumb_buffer(&base->base);
	pixman_box32_t *boxes;
	int count;

	if (!buffer->shadowed || !buffer->scanout)
		return;

	pixman_region32_t damage;
	pixman_region32_init(&damage);
	pixman_region32_intersect_rect(&damage, &base->base.damage, 0, 0,
	                               base->base.width, base->base.height);
	boxes = pixman_region32_rectangles(&damage, &count);
	while (count--) {
		int32_t y;
		size_t offset = (size_t)boxes->y1 * base->base.pitch +
		                (size_t)boxes->x1 * 4;
		size_t size = (size_t)(boxes->x2 - boxes->x1) * 4;

		for (y = boxes->y1; y < boxes->y2; ++y) {
			memcpy((uint8_t *)buffer->scanout + offset,
			       (uint8_t *)base->base.map + offset, size);
			offset += base->base.pitch;
		}
		++boxes;
	}
	pixman_region32_fini(&damage);
}

bool
buffer_unmap(struct buffer *buffer)
{
	struct dumb_buffer *dumb = dumb_buffer(&buffer->base);

	if (munmap(dumb->scanout,
	           (size_t)buffer->base.pitch * buffer->base.height)
	    == -1) {
		return false;
	}

	if (dumb->shadowed)
		free(buffer->base.map);
	dumb->scanout = NULL;
	buffer->base.map = NULL;

	return true;
}

void
buffer_destroy(struct buffer *base)
{
	struct dumb_buffer *buffer = dumb_buffer(&base->base);
	struct dumb_handle *reference = buffer->handle_ref;
	if (--reference->references == 0) {
		struct dumb_handle **link = &buffer->context->handles;
		while (*link != reference)
			link = &(*link)->next;
		*link = reference->next;
		if (reference->allocated) {
			struct drm_mode_destroy_dumb destroy_dumb = {.handle = reference->handle};
			drmIoctl(buffer->context->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_dumb);
		} else {
			struct drm_gem_close close_arg = {.handle = reference->handle};
			drmIoctl(buffer->context->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
		}
		free(reference);
	}
	free(buffer);
}
