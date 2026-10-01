/* Exercise dispatch without opening a device or creating a GPU context. */
#define WITH_DRM_GBM 1
#define WITH_DRM_INTEL 1
#define WITH_DRM_NOUVEAU 1
#include "../drm.c"
#include <assert.h>

static drmDevice device;
static drmPciDeviceInfo pci;
static struct wld_context accelerated, software;
static bool query_fails, gbm_fails;
static unsigned gbm_calls, intel_calls, nouveau_calls, dumb_calls;
const struct wld_context_impl *dumb_context_impl;

int drmGetDevice2(int fd, uint32_t flags, drmDevicePtr *out)
{
	(void)fd; (void)flags;
	*out = query_fails ? NULL : &device;
	return query_fails ? -1 : 0;
}
void drmFreeDevice(drmDevicePtr *out) { *out = NULL; }
static bool any_device(uint32_t vendor, uint32_t id) { return true; }
static bool intel_device(uint32_t vendor, uint32_t id) { return vendor == 0x8086; }
static bool nouveau_device(uint32_t vendor, uint32_t id) { return vendor == 0x10de; }
static struct wld_context *gbm_create(int fd)
{ ++gbm_calls; return gbm_fails ? NULL : &accelerated; }
static struct wld_context *intel_create(int fd) { ++intel_calls; return &accelerated; }
static struct wld_context *nouveau_create(int fd) { ++nouveau_calls; return &accelerated; }
static struct wld_context *dumb_create(int fd) { ++dumb_calls; return &software; }
const struct drm_driver gbm_drm_driver = {
	.name="gbm", .device_supported=any_device, .create_context=gbm_create,
};
const struct drm_driver intel_drm_driver = {
	.name="intel", .requires_pci=true, .device_supported=intel_device, .create_context=intel_create,
};
const struct drm_driver nouveau_drm_driver = {
	.name="nouveau", .requires_pci=true, .device_supported=nouveau_device, .create_context=nouveau_create,
};
const struct drm_driver dumb_drm_driver = {
	.name="dumb", .device_supported=any_device, .create_context=dumb_create,
};

static void reset(int bus)
{
	device = (drmDevice){ .bustype=bus };
	pci = (drmPciDeviceInfo){ .vendor_id=0x8086, .device_id=123 };
	if (bus == DRM_BUS_PCI) device.deviceinfo.pci = &pci;
	query_fails = gbm_fails = false;
	gbm_calls = intel_calls = nouveau_calls = dumb_calls = 0;
	unsetenv("WLD_DRM_DRIVER"); unsetenv("WLD_DRM_NO_GBM"); unsetenv("WLD_DRM_DUMB");
}

int main(void)
{
	reset(DRM_BUS_PLATFORM);
	assert(wld_drm_create_context(42) == &accelerated && gbm_calls == 1);
	assert(!intel_calls && !nouveau_calls && !dumb_calls);
	reset(DRM_BUS_PLATFORM); gbm_fails = true;
	assert(wld_drm_create_context(42) == &software && dumb_calls == 1);
	assert(!intel_calls && !nouveau_calls);
	reset(DRM_BUS_PLATFORM); query_fails = true;
	assert(wld_drm_create_context(42) == &accelerated && gbm_calls == 1);
	reset(DRM_BUS_PCI); gbm_fails = true;
	assert(wld_drm_create_context(42) == &accelerated && intel_calls == 1);
	assert(!nouveau_calls && !dumb_calls);
	reset(DRM_BUS_PCI); pci.vendor_id = 0x10de; gbm_fails = true;
	assert(wld_drm_create_context(42) == &accelerated && nouveau_calls == 1);
	assert(!intel_calls && !dumb_calls);
	reset(DRM_BUS_PLATFORM); setenv("WLD_DRM_NO_GBM", "1", 1);
	assert(wld_drm_create_context(42) == &software && !gbm_calls);
	reset(DRM_BUS_PLATFORM); setenv("WLD_DRM_DRIVER", "intel", 1);
	assert(wld_drm_create_context(42) == &software && !gbm_calls && !intel_calls);
	reset(DRM_BUS_PCI); setenv("WLD_DRM_DUMB", "1", 1);
	assert(wld_drm_create_context(42) == &software && !gbm_calls && dumb_calls == 1);
	puts("DRM dispatch: platform, PCI, query failure, fallback and overrides passed");
	return 0;
}
