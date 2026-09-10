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

#include <xf86drm.h>

const static struct drm_driver *drivers[] = {
#if WITH_DRM_INTEL
	&intel_drm_driver,
#endif
#if WITH_DRM_NOUVEAU
	&nouveau_drm_driver,
#endif
#if WITH_DRM_GBM
	/*
	 * Generic GBM/EGL acceleration. Last before the software fallback, so
	 * the hardware-specific drivers above keep priority where they apply.
	 */
	&gbm_drm_driver,
#endif
	&dumb_drm_driver
};

/*
 * Try each driver that claims this device, in priority order, until one
 * produces a context.
 *
 * Trying only the first match is not enough: nouveau claims every NVIDIA PCI
 * ID but supports only chipset families 0xc0 and 0xd0, so on anything newer it
 * matches, fails to create a context, and would shadow the generic GBM driver
 * behind it.
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
		if (!drivers[index]->device_supported(vendor_id, device_id))
			continue;

		DEBUG("Trying DRM driver `%s'\n", drivers[index]->name);
		if ((context = drivers[index]->create_context(fd)))
			break;

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
	return dumb_drm_driver.create_context(fd);
}

EXPORT
bool
wld_drm_is_dumb(struct wld_context *context)
{
	return context->impl == dumb_context_impl;
}
