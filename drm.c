/* wld: drm.c
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

#include "drm.h"
#include "drm-private.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>

/*
 * In priority order.
 *
 * The generic GBM/EGL backend comes first because it is the only accelerated
 * one that understands DRM format modifiers and can wait on a client's
 * fences. The hardware-specific backends below have neither: they cannot
 * enumerate modifiers, so swc advertises DRM_FORMAT_MOD_INVALID and a client
 * hands over its dma-buf with no layout information, which these backends
 * then read as if it were untiled. Current Mesa renders into tiled buffers on
 * every generation they cover, so what they blit is misinterpreted rather
 * than merely unaccelerated.
 *
 * They stay as fallbacks for systems with no usable GBM/EGL stack, where
 * clients render in software into linear buffers and the distinction does not
 * arise.
 */
static const struct drm_driver *drivers[] = {
#if WITH_DRM_GBM
	&gbm_drm_driver,
#endif
#if WITH_DRM_INTEL
	&intel_drm_driver,
#endif
#if WITH_DRM_NOUVEAU
	&nouveau_drm_driver,
#endif
	&dumb_drm_driver
};

/*
 * WLD_DRM_DRIVER=<name> restricts the search to that one driver, and
 * WLD_DRM_NO_<NAME> removes one from it. Both take the names in the table
 * above: gbm, intel, nouveau, dumb.
 *
 * Which backend claimed the device is the first thing worth varying when a
 * display is corrupt or slow, and rebuilding the library is a poor way to
 * bisect that.
 */
static bool
driver_selected(const char *name)
{
	const char *only = getenv("WLD_DRM_DRIVER");
	char variable[64];
	size_t i;
	int length;

	if (only && *only && strcmp(only, name) != 0)
		return false;

	length = snprintf(variable, sizeof variable, "WLD_DRM_NO_%s", name);
	if (length < 0 || (size_t)length >= sizeof variable)
		return true;
	for (i = 0; variable[i]; ++i)
		variable[i] = toupper((unsigned char)variable[i]);

	if (getenv(variable)) {
		fprintf(stderr, "wld: DRM backend %s disabled by %s\n", name, variable);
		return false;
	}

	return true;
}

/*
 * Try each driver that claims this device, in priority order, until one
 * produces a context.
 *
 * Trying only the first match is not enough, because a driver may claim a
 * device it cannot actually drive. GBM claims everything, so that an unusable
 * EGL stack is discovered rather than assumed, and nouveau claims every NVIDIA
 * PCI ID but supports only chipset families 0xc0 and 0xd0. Either one fails in
 * create_context, and the loop moves on to the next candidate.
 */
static struct wld_context *
create_driver_context(int fd)
{
	drmDevicePtr device = NULL;
	uint32_t vendor_id, device_id;
	struct wld_context *context = NULL;
	uint32_t index;

	if (drmGetDevice2(fd, 0, &device) != 0)
		return NULL;

	if (device->bustype != DRM_BUS_PCI || !device->deviceinfo.pci)
		goto out;

	vendor_id = device->deviceinfo.pci->vendor_id;
	device_id = device->deviceinfo.pci->device_id;

	for (index = 0; index < ARRAY_LENGTH(drivers); ++index) {
		if (!driver_selected(drivers[index]->name))
			continue;

		if (!drivers[index]->device_supported(vendor_id, device_id))
			continue;

		DEBUG("Trying DRM driver `%s'\n", drivers[index]->name);
		if ((context = drivers[index]->create_context(fd))) {
			fprintf(stderr, "wld: selected DRM backend %s\n", drivers[index]->name);
			break;
		}

		DEBUG("DRM driver `%s' did not take the device\n",
		      drivers[index]->name);
	}

out:
	drmFreeDevice(&device);
	return context;
}

EXPORT
struct wld_context *
wld_drm_create_context(int fd)
{
	struct wld_context *context;

	if (!getenv("WLD_DRM_DUMB")) {
		context = create_driver_context(fd);
		if (context)
			return context;
	}

	DEBUG("Falling back to dumb DRM driver\n");
	fprintf(stderr, "wld: falling back to software dumb DRM backend\n");
	return dumb_drm_driver.create_context(fd);
}

EXPORT
int
wld_drm_query_modifiers(struct wld_context *context, uint32_t format,
                        uint64_t *modifiers, int max)
{
	if (!context->impl->query_modifiers)
		return -1;

	return context->impl->query_modifiers(context, format, modifiers, max);
}

EXPORT
bool
wld_drm_is_dumb(struct wld_context *context)
{
	return context->impl == dumb_context_impl;
}
