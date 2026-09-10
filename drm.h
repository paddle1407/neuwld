/* wld: drm.h
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

#ifndef WLD_DRM_H
#define WLD_DRM_H

#include <stdbool.h>
#include <stdint.h>

#define WLD_DRM_ID (0x02 << 24)

enum wld_drm_object_type {
	WLD_DRM_OBJECT_HANDLE = WLD_DRM_ID,
	WLD_DRM_OBJECT_PRIME_FD,
	/**
	 * The buffer's DRM format modifier, as a uint64_t in object->u64.
	 *
	 * A caller building a KMS framebuffer needs this: a tiled buffer passed
	 * to drmModeAddFB2() without DRM_MODE_FB_MODIFIERS is interpreted as
	 * linear and rejected.
	 */
	WLD_DRM_OBJECT_MODIFIER,
	/**
	 * A dmabuf with an explicit format modifier.
	 *
	 * object.ptr points at a struct wld_dmabuf_attributes. Unlike
	 * WLD_DRM_OBJECT_PRIME_FD this carries the layout, which a tiled buffer
	 * cannot be imported correctly without.
	 */
	WLD_DRM_OBJECT_DMABUF,
};

struct wld_dmabuf_attributes {
	int fd;
	uint32_t offset;
	uint32_t pitch;
	uint64_t modifier;
};

enum wld_drm_flags {
	WLD_DRM_FLAG_SCANOUT = 0x1,
	WLD_DRM_FLAG_TILED = 0x2
};

/**
 * Create a new WLD context from an opened DRM device file descriptor.
 */
struct wld_context *wld_drm_create_context(int fd);

bool wld_drm_is_dumb(struct wld_context *context);

/**
 * Fill 'modifiers' with the DRM format modifiers this context can import for
 * 'format'. Returns the number written, or -1 if the backend cannot say.
 */
int wld_drm_query_modifiers(struct wld_context *context, uint32_t format,
                            uint64_t *modifiers, int max);

#endif
