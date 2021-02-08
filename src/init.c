#include "../include/init.h"

#include <stdlib.h>
#include <stdarg.h>

Initializer *Initializer_Create(InitializerType type, ...) {
    Initializer *init;
    va_list ap;

    init = calloc(1, sizeof(*init));
    init->type = type;

    va_start(ap, type);
    switch (type) {
    case INIT_CONST:
        init->const_u1 = va_arg(ap, double);
        init->const_u2 = va_arg(ap, double);
        init->const_u3 = va_arg(ap, double);
        init->const_p = va_arg(ap, double);
        break;
    case INIT_FUNC:
        init->func_u1 = va_arg(ap, InitializerFunc);
        init->func_u2 = va_arg(ap, InitializerFunc);
        init->func_u3 = va_arg(ap, InitializerFunc);
        init->func_p = va_arg(ap, InitializerFunc);
        break;
    case INIT_NETCDF:
        init->netcdf_filename = va_arg(ap, const char *);
        break;
    }
    va_end(ap);

    return init;
}

void Initializer_Destroy(Initializer *init) {
    free(init);
}
