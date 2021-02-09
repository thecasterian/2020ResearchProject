#ifndef UTILS_H
#define UTILS_H

#define VA_GENERIC(_1, _2, _3, _4, x, ...) x

/* Access to an element of an 1d coallocated array. */
#define c1e(a, i) ((a)[(i)+3])
/* Access to an element of an 3d coallocated grid. */
#define c3e(a, i, j, k) ((a)[(mesh->Ny_l+6)*(mesh->Nz_l+6)*((i)+3) \
                             + (mesh->Nz_l+6)*((j)+3) + (k)+3])

#define e(...) VA_GENERIC(__VA_ARGS__, c3e, NULL, c1e, NULL)(__VA_ARGS__)

#define FOR_U(i, j, k) \
    for (int i = 0; i < mesh->Nx_l; i++) \
        for (int j = 0; j < mesh->Ny_l; j++) \
            for (int k = 0; k < mesh->Nz_l; k++)


#define FOR_U_ALL(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_F1_ALL(i, j, k) \
    for (int i = -2; i <= mesh->Nx_l+2; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_F2_ALL(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -2; j <= mesh->Ny_l+2; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_F3_ALL(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -2; k <= mesh->Nz_l+2; k++)

#define FOR_OUTER_WEST(i, j, k) \
    for (int i = -3; i <= -1; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_OUTER_EAST(i, j, k) \
    for (int i = mesh->Nx_l; i <= mesh->Nx_l+2; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_OUTER_SOUTH(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j <= -1; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_OUTER_NORTH(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = mesh->Ny_l; j <= mesh->Ny_l+2; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_OUTER_DOWN(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k <= -1; k++)
#define FOR_OUTER_UP(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = mesh->Nz_l; k <= mesh->Nz_l+2; k++)

#define FOR_INNER_WEST(i, j, k) \
    for (int i = 0; i <= 2; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_INNER_EAST(i, j, k) \
    for (int i = mesh->Nx_l-3; i <= mesh->Nx_l-1; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_INNER_SOUTH(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = 0; j <= 2; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_INNER_NORTH(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = mesh->Ny_l-3; j <= mesh->Ny_l-1; j++) \
            for (int k = -3; k < mesh->Nz_l+3; k++)
#define FOR_INNER_DOWN(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = 0; k <= 2; k++)
#define FOR_INNER_UP(i, j, k) \
    for (int i = -3; i < mesh->Nx_l+3; i++) \
        for (int j = -3; j < mesh->Ny_l+3; j++) \
            for (int k = mesh->Nz_l-3; k <= mesh->Nz_l-1; k++)

#define max(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a > _b ? _a : _b;})
#define min(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a < _b ? _a : _b;})
#define swap(a, b) do {typeof(a) tmp = a; a = b; b = tmp;} while (0)

#define UNUSED __attribute__((unused))

int lower_bound(const int len, const double arr[], const double val);
int upper_bound(const int len, const double arr[], const double val);

int divceil(int num, int den);

/*
           i-2     i-1      i      i+1     i+2     i+3
           www      ww      w       e       ee     eee
    +-------+-------+-------+-------+-------+-------+-------+
    |   *   |   *   |   *   |   *   |   *   |   *   |   *   |
    |  WWW  |   WW  |   W   |   P   |   E   |   EE  |  EEE  |
    +-------+-------+-------+-------+-------+-------+-------+
       i-3     i-2     i-1      i      i+1     i+2     i+3
*/

#define u1P   e(vars->u1, i, j, k)
#define u1WWW e(vars->u1, i-3, j, k)
#define u1WW  e(vars->u1, i-2, j, k)
#define u1W   e(vars->u1, i-1, j, k)
#define u1E   e(vars->u1, i+1, j, k)
#define u1EE  e(vars->u1, i+2, j, k)
#define u1EEE e(vars->u1, i+3, j, k)
#define u1SSS e(vars->u1, i, j-3, k)
#define u1SS  e(vars->u1, i, j-2, k)
#define u1S   e(vars->u1, i, j-1, k)
#define u1N   e(vars->u1, i, j+1, k)
#define u1NN  e(vars->u1, i, j+2, k)
#define u1NNN e(vars->u1, i, j+3, k)
#define u1DDD e(vars->u1, i, j, k-3)
#define u1DD  e(vars->u1, i, j, k-2)
#define u1D   e(vars->u1, i, j, k-1)
#define u1U   e(vars->u1, i, j, k+1)
#define u1UU  e(vars->u1, i, j, k+2)
#define u1UUU e(vars->u1, i, j, k+3)

#define u2P   e(vars->u2, i, j, k)
#define u2WWW e(vars->u2, i-3, j, k)
#define u2WW  e(vars->u2, i-2, j, k)
#define u2W   e(vars->u2, i-1, j, k)
#define u2E   e(vars->u2, i+1, j, k)
#define u2EE  e(vars->u2, i+2, j, k)
#define u2EEE e(vars->u2, i+3, j, k)
#define u2SSS e(vars->u2, i, j-3, k)
#define u2SS  e(vars->u2, i, j-2, k)
#define u2S   e(vars->u2, i, j-1, k)
#define u2N   e(vars->u2, i, j+1, k)
#define u2NN  e(vars->u2, i, j+2, k)
#define u2NNN e(vars->u2, i, j+3, k)
#define u2DDD e(vars->u2, i, j, k-3)
#define u2DD  e(vars->u2, i, j, k-2)
#define u2D   e(vars->u2, i, j, k-1)
#define u2U   e(vars->u2, i, j, k+1)
#define u2UU  e(vars->u2, i, j, k+2)
#define u2UUU e(vars->u2, i, j, k+3)

#define u3P   e(vars->u3, i, j, k)
#define u3WWW e(vars->u3, i-3, j, k)
#define u3WW  e(vars->u3, i-2, j, k)
#define u3W   e(vars->u3, i-1, j, k)
#define u3E   e(vars->u3, i+1, j, k)
#define u3EE  e(vars->u3, i+2, j, k)
#define u3EEE e(vars->u3, i+3, j, k)
#define u3SSS e(vars->u3, i, j-3, k)
#define u3SS  e(vars->u3, i, j-2, k)
#define u3S   e(vars->u3, i, j-1, k)
#define u3N   e(vars->u3, i, j+1, k)
#define u3NN  e(vars->u3, i, j+2, k)
#define u3NNN e(vars->u3, i, j+3, k)
#define u3DDD e(vars->u3, i, j, k-3)
#define u3DD  e(vars->u3, i, j, k-2)
#define u3D   e(vars->u3, i, j, k-1)
#define u3U   e(vars->u3, i, j, k+1)
#define u3UU  e(vars->u3, i, j, k+2)
#define u3UUU e(vars->u3, i, j, k+3)

#define F1ww e(vars->F1, i-1, j, k)
#define F1w  e(vars->F1, i  , j, k)
#define F1e  e(vars->F1, i+1, j, k)
#define F1ee e(vars->F1, i+2, j, k)

#define F2ss e(vars->F2, i, j-1, k)
#define F2s  e(vars->F2, i, j  , k)
#define F2n  e(vars->F2, i, j+1, k)
#define F2nn e(vars->F2, i, j+2, k)

#define F3dd e(vars->F3, i, j, k-1)
#define F3d  e(vars->F3, i, j, k  )
#define F3u  e(vars->F3, i, j, k+1)
#define F3uu e(vars->F3, i, j, k+2)

#endif
