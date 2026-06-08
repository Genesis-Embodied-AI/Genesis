// nyx_headless_shim.c — make Nyx's Vulkan renderer work on a display-less
// NVIDIA node (e.g. H20-3e, headless datacenter GPU).
//
// Two independent problems are patched here, both via LD_PRELOAD interposition:
//
//   1. GLFW: Nyx calls glfwInit() with no platform hint. The bundled GLFW is
//      X11-only at runtime selection; with no DISPLAY it hard-fails instead of
//      falling back to its (compiled-in) Null platform. We force
//      GLFW_PLATFORM_NULL so glfwInit() succeeds and requests zero WSI
//      instance extensions.
//
//   2. Vulkan: the actual segfault is inside vkCreateDevice in the NVIDIA
//      driver (libGLX_nvidia -> libnvidia-eglcore), triggered by enabling WSI
//      extensions (VK_KHR_swapchain on the device, VK_KHR_*_surface on the
//      instance) on a node with no display/present support. For an offscreen
//      path tracer that reads frames back over CUDA interop, none of these are
//      needed. We intercept vkCreateInstance / vkCreateDevice and strip the
//      WSI extensions from the enabled-extension list before delegating.
//
// Build:
//   gcc -shared -fPIC -O2 nyx_headless_shim.c -o nyx_headless_shim.so -ldl
// Use:
//   LD_PRELOAD=/path/nyx_headless_shim.so python3 your_nyx_script.py
//
// Env knobs:
//   NYX_GLFW_LIB             override path to bundled libglfw .so
//   NYX_HEADLESS_SHIM_QUIET  silence the per-call notices

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----------------------- GLFW null-platform forcing ----------------------- */

#define GLFW_PLATFORM       0x00050003
#define GLFW_PLATFORM_NULL  0x00060005
#define DEFAULT_GLFW_LIB \
    "/usr/local/lib/python3.12/dist-packages/gs_nyx.libs/libglfw-e1f5b16c.so.3.4"

static int  (*real_glfwInit)(void) = NULL;
static void (*real_glfwInitHint)(int, int) = NULL;

static int quiet(void) { return getenv("NYX_HEADLESS_SHIM_QUIET") != NULL; }

static void resolve_glfw(void) {
    if (real_glfwInit && real_glfwInitHint) return;
    if (!real_glfwInit)     real_glfwInit     = (int(*)(void))      dlsym(RTLD_NEXT, "glfwInit");
    if (!real_glfwInitHint) real_glfwInitHint = (void(*)(int,int))  dlsym(RTLD_NEXT, "glfwInitHint");
    if (!real_glfwInit || !real_glfwInitHint) {
        const char *path = getenv("NYX_GLFW_LIB");
        if (!path || !*path) path = DEFAULT_GLFW_LIB;
        void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            if (!real_glfwInit)     real_glfwInit     = (int(*)(void))     dlsym(h, "glfwInit");
            if (!real_glfwInitHint) real_glfwInitHint = (void(*)(int,int)) dlsym(h, "glfwInitHint");
        } else {
            fprintf(stderr, "[nyx-shim] dlopen(%s) failed: %s\n", path, dlerror());
        }
    }
}

int glfwInit(void) {
    resolve_glfw();
    if (!quiet()) fprintf(stderr, "[nyx-shim] forcing GLFW_PLATFORM_NULL (headless)\n");
    if (real_glfwInitHint) real_glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
    if (!real_glfwInit) { fprintf(stderr, "[nyx-shim] ERROR: no real glfwInit\n"); return 0; }
    return real_glfwInit();
}

/* ------------------------ Vulkan WSI-extension strip ---------------------- */
/* Minimal Vulkan typedefs (avoid a vulkan.h build dependency). */

typedef int32_t  VkResult;
typedef void*    VkInstance;
typedef void*    VkPhysicalDevice;
typedef void*    VkDevice;
typedef void*    VkAllocationCallbacks_p;

typedef struct {
    int32_t            sType;
    const void        *pNext;
    uint32_t           flags;
    const void        *pApplicationInfo;
    uint32_t           enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t           enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
    int32_t            sType;
    const void        *pNext;
    uint32_t           flags;
    uint32_t           queueCreateInfoCount;
    const void        *pQueueCreateInfos;
    uint32_t           enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t           enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const void        *pEnabledFeatures;
} VkDeviceCreateInfo;

/* WSI / windowing extensions that drag the NVIDIA driver into the
 * libnvidia-eglcore present path. Safe to drop for offscreen CUDA-interop
 * rendering. */
static int is_wsi_ext(const char *n) {
    static const char *wsi[] = {
        "VK_KHR_swapchain",
        "VK_KHR_surface",
        "VK_KHR_xlib_surface",
        "VK_KHR_xcb_surface",
        "VK_KHR_wayland_surface",
        "VK_KHR_display",
        "VK_KHR_display_swapchain",
        "VK_EXT_display_surface_counter",
        "VK_EXT_display_control",
        "VK_KHR_get_surface_capabilities2",
        "VK_KHR_incremental_present",
        "VK_GOOGLE_display_timing",
        NULL,
    };
    for (int i = 0; wsi[i]; i++)
        if (strcmp(n, wsi[i]) == 0) return 1;
    return 0;
}

/* Return a filtered copy of the extension name array (caller frees both the
 * array and the resulting count via *out_count). Never fails destructively:
 * on alloc failure we leave the original list intact. */
static const char **filter_exts(const char *const *in, uint32_t in_n,
                                uint32_t *out_n, const char *which) {
    if (!quiet()) {
        fprintf(stderr, "[nyx-shim] %s called with %u extension(s):", which, in_n);
        for (uint32_t i = 0; i < in_n; i++) fprintf(stderr, " %s", in[i]);
        fprintf(stderr, "\n");
    }
    if (in_n == 0 || in == NULL) { *out_n = in_n; return (const char **) in; }
    const char **out = (const char **) malloc(sizeof(char *) * in_n);
    if (!out) { *out_n = in_n; return (const char **) in; }
    uint32_t k = 0;
    for (uint32_t i = 0; i < in_n; i++) {
        if (is_wsi_ext(in[i])) {
            if (!quiet())
                fprintf(stderr, "[nyx-shim] %s: dropping WSI extension %s\n", which, in[i]);
            continue;
        }
        out[k++] = in[i];
    }
    *out_n = k;
    return out;
}

static VkResult (*real_vkCreateInstance)(const VkInstanceCreateInfo*, VkAllocationCallbacks_p, VkInstance*) = NULL;
static VkResult (*real_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, VkAllocationCallbacks_p, VkDevice*) = NULL;

/* libvulkan may load after this preload, so RTLD_NEXT can return NULL. Fall
 * back to dlopen'ing the loader explicitly. */
static void *vk_handle(void) {
    static void *h = NULL;
    if (!h) {
        const char *p = getenv("NYX_VULKAN_LIB");
        h = dlopen((p && *p) ? p : "libvulkan.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_GLOBAL);
    }
    return h;
}

static void *resolve_vk(const char *name) {
    void *fn = dlsym(RTLD_NEXT, name);
    if (!fn) { void *h = vk_handle(); if (h) fn = dlsym(h, name); }
    return fn;
}

VkResult vkCreateInstance(const VkInstanceCreateInfo *ci,
                          VkAllocationCallbacks_p alloc, VkInstance *out) {
    if (!real_vkCreateInstance)
        real_vkCreateInstance = resolve_vk("vkCreateInstance");
    if (!real_vkCreateInstance) {
        fprintf(stderr, "[nyx-shim] ERROR: cannot resolve real vkCreateInstance\n");
        return -3; /* VK_ERROR_INITIALIZATION_FAILED */
    }
    VkInstanceCreateInfo patched = *ci;
    uint32_t n = 0;
    const char **filtered = filter_exts(ci->ppEnabledExtensionNames,
                                        ci->enabledExtensionCount, &n, "vkCreateInstance");
    patched.ppEnabledExtensionNames = filtered;
    patched.enabledExtensionCount   = n;
    VkResult r = real_vkCreateInstance(&patched, alloc, out);
    if (filtered != ci->ppEnabledExtensionNames) free((void *) filtered);
    return r;
}

VkResult vkCreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                        VkAllocationCallbacks_p alloc, VkDevice *out) {
    if (!real_vkCreateDevice)
        real_vkCreateDevice = resolve_vk("vkCreateDevice");
    if (!real_vkCreateDevice) {
        fprintf(stderr, "[nyx-shim] ERROR: cannot resolve real vkCreateDevice\n");
        return -3;
    }
    VkDeviceCreateInfo patched = *ci;
    uint32_t n = 0;
    const char **filtered = filter_exts(ci->ppEnabledExtensionNames,
                                        ci->enabledExtensionCount, &n, "vkCreateDevice");
    patched.ppEnabledExtensionNames = filtered;
    patched.enabledExtensionCount   = n;
    VkResult r = real_vkCreateDevice(pd, &patched, alloc, out);
    if (filtered != ci->ppEnabledExtensionNames) free((void *) filtered);
    return r;
}

/* ---- diagnostic interception of memory alloc / bind (no behavior change) ---- */
typedef uint64_t VkBuffer;       /* non-dispatchable handle */
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkDeviceSize;

static VkResult (*real_vkAllocateMemory)(VkDevice, const void*, VkAllocationCallbacks_p, VkDeviceMemory*) = NULL;
static VkResult (*real_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) = NULL;

/* VkMemoryAllocateInfo layout (sType,pNext,allocationSize,memoryTypeIndex). */
typedef struct {
    int32_t       sType;
    const void   *pNext;
    VkDeviceSize  allocationSize;
    uint32_t      memoryTypeIndex;
} VkMemoryAllocateInfo;

VkResult vkAllocateMemory(VkDevice dev, const void *info,
                          VkAllocationCallbacks_p alloc, VkDeviceMemory *mem) {
    if (!real_vkAllocateMemory) real_vkAllocateMemory = resolve_vk("vkAllocateMemory");
    const VkMemoryAllocateInfo *ai = (const VkMemoryAllocateInfo *) info;

    /* Nyx computes a zero-byte allocationSize for (at least) its external-memory
     * interop buffer when the scene is trivial. allocationSize==0 is invalid
     * Vulkan usage; the NVIDIA driver returns VK_ERROR_OUT_OF_DEVICE_MEMORY and
     * Nyx then binds the NULL handle and segfaults. Bump a zero size up to a
     * small non-zero floor so the allocation (and the subsequent bind) succeed. */
    VkMemoryAllocateInfo patched;
    if (ai && ai->allocationSize == 0) {
        patched = *ai;
        patched.allocationSize = 256; /* >0, NVIDIA aligns up internally */
        info = &patched;
        if (!quiet())
            fprintf(stderr, "[nyx-shim] vkAllocateMemory: bumping size 0 -> 256 "
                            "(typeIdx=%u, external-memory interop)\n", ai->memoryTypeIndex);
    }

    VkResult r = real_vkAllocateMemory(dev, info, alloc, mem);
    if (!quiet())
        fprintf(stderr, "[nyx-shim] vkAllocateMemory size=%llu typeIdx=%u pNext=%p -> r=%d mem=0x%llx\n",
                (unsigned long long)(ai ? ai->allocationSize : 0),
                ai ? ai->memoryTypeIndex : 0, ai ? ai->pNext : 0,
                r, (unsigned long long)(mem ? *mem : 0));
    return r;
}

VkResult vkBindBufferMemory(VkDevice dev, VkBuffer buf,
                            VkDeviceMemory mem, VkDeviceSize off) {
    if (!real_vkBindBufferMemory) real_vkBindBufferMemory = resolve_vk("vkBindBufferMemory");
    if (mem == 0) {
        /* Nyx does not check vkAllocateMemory's result and binds a NULL
         * VkDeviceMemory; the NVIDIA driver dereferences it and crashes.
         * Return the error cleanly instead so the failure is at least
         * catchable rather than a SIGSEGV. */
        fprintf(stderr, "[nyx-shim] vkBindBufferMemory: NULL memory handle "
                        "(upstream alloc failed) -> returning VK_ERROR_OUT_OF_DEVICE_MEMORY\n");
        return -2; /* VK_ERROR_OUT_OF_DEVICE_MEMORY */
    }
    if (!quiet())
        fprintf(stderr, "[nyx-shim] vkBindBufferMemory dev=%p buf=0x%llx mem=0x%llx off=%llu\n",
                dev, (unsigned long long)buf, (unsigned long long)mem,
                (unsigned long long)off);
    return real_vkBindBufferMemory(dev, buf, mem, off);
}
