#ifndef IBM3D_INIT_H
#define IBM3D_INIT_H

typedef enum _initializer_type {
    INIT_CONST,
    INIT_FUNC,
    INIT_NETCDF,
} InitializerType;

typedef double (*InitializerFunc)(double, double, double);

typedef struct _initializer {
    InitializerType type;
    union {
        struct {
            double const_u1, const_u2, const_u3, const_p;
        };
        struct {
            InitializerFunc func_u1, func_u2, func_u3, func_p;
        };
        const char *netcdf_filename;
    };
} Initializer;

Initializer *Initializer_Create(InitializerType type, ...);
void Initializer_Destroy(Initializer *init);

#endif