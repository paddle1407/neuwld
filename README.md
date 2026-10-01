neuwld
===
neuwld is the drawing library used by
[charaWC](https://github.com/paddle1407/charawc) and
[neuswc](https://github.com/paddle1407/neuswc). It is a fork of Michael
Forney's [wld](https://github.com/michaelforney/wld), with Pixman software
rendering and DRM backends for GBM/EGL/GLES, Intel, Nouveau and dumb buffers.

Installing
==========
Use Meson 1.8 or newer, Ninja, a C23 compiler and pkg-config. Fontconfig,
FreeType and Pixman are required; Wayland, libdrm and the GPU libraries are
selected by the build options.

```Bash
meson setup build
ninja -C build
meson install -C build
```

For a CPU-only build, configure with `-Ddrm=disabled -Dwayland=disabled`.
GBM can drive PCI and platform GPUs; the legacy Intel and Nouveau backends
require PCI identification. `WLD_DRM_DRIVER=gbm` restricts driver selection,
`WLD_DRM_NO_GBM=1` disables GBM, and `WLD_DRM_DUMB=1` forces software DRM.

The DRM dispatch regression check is registered with Meson. Building compiles
the check without executing it; run it explicitly with
`meson test -C build --print-errorlogs`. It uses mocked devices and does not
open a GPU or start a compositor.
