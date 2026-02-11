// Quick test: verify NOE runtime connects to NPU
// Build: gcc -o test_noe test_noe.c -ldl
// Run:   LD_LIBRARY_PATH=/usr/share/cix/lib ./test_noe

#include <stdio.h>
#include <dlfcn.h>

typedef int (*noe_init_fn)(void**);

int main() {
    void *lib = dlopen("libnoe.so", RTLD_NOW);
    if (!lib) {
        printf("Failed to load libnoe.so: %s\n", dlerror());
        return 1;
    }

    noe_init_fn init = dlsym(lib, "noe_init_context");
    if (!init) {
        printf("Failed to find noe_init_context: %s\n", dlerror());
        dlclose(lib);
        return 1;
    }

    void *ctx = NULL;
    int ret = init(&ctx);
    printf("noe_init_context returned: %d (0=success), ctx=%p\n", ret, ctx);

    dlclose(lib);
    return ret;
}
