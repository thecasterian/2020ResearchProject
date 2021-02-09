#include "../include/vars.h"

#include "../include/mesh.h"
#include "../include/utils.h"
#include "../include/init.h"
#include <stdlib.h>
#include <mpi.h>
#include <stdarg.h>

typedef enum {
    PLUS_X,
    MINUS_X,
    PLUS_Y,
    MINUS_Y,
    PLUS_Z,
    MINUS_Z,
} Axis;

static void Exchg(FlowVars *vars, PartMesh *mesh, int n, ...);

FlowVars *FlowVars_Create(PartMesh *mesh) {
    FlowVars *vars;
    int sz = (mesh->Nx_l+6) * (mesh->Ny_l+6) * (mesh->Nz_l+6);

    vars = calloc(1, sizeof(*vars));

    vars->u1      = calloc(sz, sizeof(double));
    vars->u1_next = calloc(sz, sizeof(double));
    vars->u2      = calloc(sz, sizeof(double));
    vars->u2_next = calloc(sz, sizeof(double));
    vars->u3      = calloc(sz, sizeof(double));
    vars->u3_next = calloc(sz, sizeof(double));

    vars->F1      = calloc(sz, sizeof(double));
    vars->F1_next = calloc(sz, sizeof(double));
    vars->F2      = calloc(sz, sizeof(double));
    vars->F2_next = calloc(sz, sizeof(double));
    vars->F3      = calloc(sz, sizeof(double));
    vars->F3_next = calloc(sz, sizeof(double));

    vars->p       = calloc(sz, sizeof(double));
    vars->p_next  = calloc(sz, sizeof(double));
    vars->delta_p = calloc(sz, sizeof(double));

    vars->H1      = calloc(sz, sizeof(double));
    vars->H1_prev = calloc(sz, sizeof(double));
    vars->H2      = calloc(sz, sizeof(double));
    vars->H2_prev = calloc(sz, sizeof(double));
    vars->H3      = calloc(sz, sizeof(double));
    vars->H3_prev = calloc(sz, sizeof(double));

    vars->x_exchg_size = 12*(mesh->Ny_l+6)*(mesh->Nz_l+6);
    vars->y_exchg_size = 12*(mesh->Nx_l+6)*(mesh->Nz_l+6);
    vars->z_exchg_size = 12*(mesh->Nx_l+6)*(mesh->Ny_l+6);

    vars->x_exchg = calloc(vars->x_exchg_size, sizeof(double));
    vars->y_exchg = calloc(vars->y_exchg_size, sizeof(double));
    vars->z_exchg = calloc(vars->z_exchg_size, sizeof(double));

    return vars;
}

void FlowVars_Initialize(FlowVars *vars, PartMesh *mesh, Initializer *init) {
    int ig, jg, kg;
    double x, y, z;

    switch (init->type) {
    case INIT_CONST:
        FOR_U_ALL (i, j, k) {
            e(vars->u1, i, j, k) = init->const_u1;
            e(vars->u2, i, j, k) = init->const_u2;
            e(vars->u3, i, j, k) = init->const_u3;
            e(vars->p, i, j, k) = init->const_p;
        }
        break;
    case INIT_FUNC:
        FOR_U_ALL (i, j, k) {
            ig = i + mesh->ilower;
            jg = j + mesh->jlower;
            kg = k + mesh->klower;

            x = (e(mesh->xf, ig) + e(mesh->xf, ig+1)) / 2;
            y = (e(mesh->yf, jg) + e(mesh->yf, jg+1)) / 2;
            z = (e(mesh->zf, kg) + e(mesh->zf, kg+1)) / 2;

            e(vars->u1, i, j, k) = init->func_u1(x, y, z);
            e(vars->u2, i, j, k) = init->func_u2(x, y, z);
            e(vars->u3, i, j, k) = init->func_u3(x, y, z);
            e(vars->p, i, j, k) = init->func_p(x, y, z);
        }
        break;
    case INIT_NETCDF:
        // TODO: read netcdf reader.
        break;
    }
}

void FlowVars_UpdateOuter(FlowVars *vars, PartMesh *mesh) {
    int cnt;

    if (mesh->ri == 0) {
        if (mesh->Px == 1) {
            FOR_OUTER_WEST (i, j, k) {
                e(vars->u1, i, j, k) = e(vars->u1, i+mesh->Nx_l, j, k);
                e(vars->u2, i, j, k) = e(vars->u2, i+mesh->Nx_l, j, k);
                e(vars->u3, i, j, k) = e(vars->u3, i+mesh->Nx_l, j, k);
                e(vars->p, i, j, k) = e(vars->p, i+mesh->Nx_l, j, k);
            }
        } else {
            cnt = 0;
            FOR_INNER_WEST (i, j, k) {
                vars->x_exchg[cnt++] = e(vars->u1, i, j, k);
                vars->x_exchg[cnt++] = e(vars->u2, i, j, k);
                vars->x_exchg[cnt++] = e(vars->u3, i, j, k);
                vars->x_exchg[cnt++] = e(vars->p, i, j, k);
            }
            MPI_Send(vars->x_exchg, vars->x_exchg_size, MPI_DOUBLE,
                     mesh->prevx, 0, MPI_COMM_WORLD);
            MPI_Recv(vars->x_exchg, vars->x_exchg_size, MPI_DOUBLE,
                     mesh->prevx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            FOR_OUTER_WEST (i, j, k) {
                e(vars->u1, i, j, k) = vars->x_exchg[cnt++];
                e(vars->u2, i, j, k) = vars->x_exchg[cnt++];
                e(vars->u3, i, j, k) = vars->x_exchg[cnt++];
                e(vars->p, i, j, k) = vars->x_exchg[cnt++];
            }
        }
    }

    if (mesh->ri == mesh->Px-1) {
        if (mesh->Px == 1) {
            FOR_OUTER_EAST (i, j, k) {
                e(vars->u1, i, j, k) = e(vars->u1, i-mesh->Nx_l, j, k);
                e(vars->u2, i, j, k) = e(vars->u2, i-mesh->Nx_l, j, k);
                e(vars->u3, i, j, k) = e(vars->u3, i-mesh->Nx_l, j, k);
                e(vars->p, i, j, k) = e(vars->p, i-mesh->Nx_l, j, k);
            }
        } else {
            MPI_Recv(vars->x_exchg, vars->x_exchg_size, MPI_DOUBLE,
                     mesh->nextx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            FOR_OUTER_EAST (i, j, k) {
                e(vars->u1, i, j, k) = vars->x_exchg[cnt++];
                e(vars->u2, i, j, k) = vars->x_exchg[cnt++];
                e(vars->u3, i, j, k) = vars->x_exchg[cnt++];
                e(vars->p, i, j, k) = vars->x_exchg[cnt++];
            }
            cnt = 0;
            FOR_INNER_EAST (i, j, k) {
                vars->x_exchg[cnt++] = e(vars->u1, i, j, k);
                vars->x_exchg[cnt++] = e(vars->u2, i, j, k);
                vars->x_exchg[cnt++] = e(vars->u3, i, j, k);
                vars->x_exchg[cnt++] = e(vars->p, i, j, k);
            }
            MPI_Send(vars->x_exchg, vars->x_exchg_size, MPI_DOUBLE,
                     mesh->nextx, 0, MPI_COMM_WORLD);
        }
    }

    if (mesh->rj == 0) {
        if (mesh->Py == 1) {
            FOR_OUTER_SOUTH (i, j, k) {
                e(vars->u1, i, j, k) = e(vars->u1, i, j+mesh->Ny_l, k);
                e(vars->u2, i, j, k) = e(vars->u2, i, j+mesh->Ny_l, k);
                e(vars->u3, i, j, k) = e(vars->u3, i, j+mesh->Ny_l, k);
                e(vars->p, i, j, k) = e(vars->p, i, j+mesh->Ny_l, k);
            }
        } else {
            cnt = 0;
            FOR_INNER_SOUTH (i, j, k) {
                vars->y_exchg[cnt++] = e(vars->u1, i, j, k);
                vars->y_exchg[cnt++] = e(vars->u2, i, j, k);
                vars->y_exchg[cnt++] = e(vars->u3, i, j, k);
                vars->y_exchg[cnt++] = e(vars->p, i, j, k);
            }
            MPI_Send(vars->y_exchg, vars->y_exchg_size, MPI_DOUBLE,
                     mesh->prevy, 0, MPI_COMM_WORLD);
            MPI_Recv(vars->y_exchg, vars->y_exchg_size, MPI_DOUBLE,
                     mesh->prevy, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            FOR_OUTER_SOUTH (i, j, k) {
                e(vars->u1, i, j, k) = vars->y_exchg[cnt++];
                e(vars->u2, i, j, k) = vars->y_exchg[cnt++];
                e(vars->u3, i, j, k) = vars->y_exchg[cnt++];
                e(vars->p, i, j, k) = vars->y_exchg[cnt++];
            }
        }
    }

    if (mesh->rj == mesh->Py-1) {
        if (mesh->Py == 1) {
            FOR_OUTER_NORTH (i, j, k) {
                e(vars->u1, i, j, k) = e(vars->u1, i, j-mesh->Ny_l, k);
                e(vars->u2, i, j, k) = e(vars->u2, i, j-mesh->Ny_l, k);
                e(vars->u3, i, j, k) = e(vars->u3, i, j-mesh->Ny_l, k);
                e(vars->p, i, j, k) = e(vars->p, i, j-mesh->Ny_l, k);
            }
        } else {
            MPI_Recv(vars->y_exchg, vars->y_exchg_size, MPI_DOUBLE,
                     mesh->nexty, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cnt = 0;
            FOR_OUTER_NORTH (i, j, k) {
                e(vars->u1, i, j, k) = vars->y_exchg[cnt++];
                e(vars->u2, i, j, k) = vars->y_exchg[cnt++];
                e(vars->u3, i, j, k) = vars->y_exchg[cnt++];
                e(vars->p, i, j, k) = vars->y_exchg[cnt++];
            }
            cnt = 0;
            FOR_INNER_NORTH (i, j, k) {
                vars->y_exchg[cnt++] = e(vars->u1, i, j, k);
                vars->y_exchg[cnt++] = e(vars->u2, i, j, k);
                vars->y_exchg[cnt++] = e(vars->u3, i, j, k);
                vars->y_exchg[cnt++] = e(vars->p, i, j, k);
            }
            MPI_Send(vars->y_exchg, vars->y_exchg_size, MPI_DOUBLE,
                     mesh->nexty, 0, MPI_COMM_WORLD);
        }
    }

    if (mesh->rk == 0) {
        FOR_OUTER_DOWN (i, j, k) {
            e(vars->u1, i, j, k) = -e(vars->u1, i, j, -k-1);
            e(vars->u2, i, j, k) = -e(vars->u2, i, j, -k-1);
            e(vars->u3, i, j, k) = -e(vars->u3, i, j, -k-1);
            e(vars->p, i, j, k) = e(vars->p, i, j, -k-1);
        }
    }

    if (mesh->rk == mesh->Px-1) {
        FOR_OUTER_UP (i, j, k) {
            e(vars->u1, i, j, k) = -e(vars->u1, i, j, -k+2*mesh->Nz_l-1);
            e(vars->u2, i, j, k) = -e(vars->u2, i, j, -k+2*mesh->Nz_l-1);
            e(vars->u3, i, j, k) = -e(vars->u3, i, j, -k+2*mesh->Nz_l-1);
            e(vars->p, i, j, k) = e(vars->p, i, j, -k+2*mesh->Nz_l-1);
        }
    }
}

void FlowVars_AdjExchg(FlowVars *vars, PartMesh *mesh) {
    Exchg(vars, mesh, 4, vars->u1, vars->u2, vars->u3, vars->p);
}

void FlowVars_InterpF(FlowVars *vars, PartMesh *mesh) {
    FOR_F1_ALL (i, j, k)
        e(vars->F1, i, j, k) = 1/(e(mesh->deta_dy_c, j)*e(mesh->dzeta_dz_c, k))
                               * (9./8 * (u1P+u1W)/2 - 1./8 * (u1E+u1WW)/2);
    FOR_F2_ALL (i, j, k)
        e(vars->F2, i, j, k) = 1/(e(mesh->dxi_dx_c, i)*e(mesh->dzeta_dz_c, k))
                               * (9./8 * (u2P+u2S)/2 - 1./8 * (u2N+u2SS)/2);
    FOR_F3_ALL (i, j, k)
        e(vars->F3, i, j, k) = 1/(e(mesh->dxi_dx_c, i)*e(mesh->deta_dy_c, j))
                               * (9./8 * (u3P+u3D)/2 - 1./8 * (u3U+u3DD)/2);
}

void FlowVars_CalcH(FlowVars *vars, PartMesh *mesh) {
    double dF1u1_dxi, dF2u1_deta, dF3u1_dzeta;
    double dF1u2_dxi, dF2u2_deta, dF3u2_dzeta;
    double dF1u3_dxi, dF2u3_deta, dF3u3_dzeta;

    FOR_U (i, j, k) {
        dF1u1_dxi = (9./8) * (F1e*(u1E+u1P)/2 - F1w*(u1P+u1W)/2)
                    - (1./24) * (F1ee*(u1EEE+u1P)/2 - F1ww*(u1P+u1WWW)/2);
        dF2u1_deta = (9./8) * (F2n*(u1N+u1P)/2 - F2s*(u1P+u1S)/2)
                     - (1./24) * (F2nn*(u1NNN+u1P)/2 - F2ss*(u1P+u1SSS)/2);
        dF3u1_dzeta = (9./8) * (F3u*(u1U+u1P)/2 - F3d*(u1P+u1D)/2)
                      - (1./24) * (F3uu*(u1UUU+u1P)/2 - F3dd*(u1P+u1DDD)/2);

        dF1u2_dxi = (9./8) * (F1e*(u2E+u2P)/2 - F1w*(u2P+u2W)/2)
                    - (1./24) * (F1ee*(u2EEE+u2P)/2 - F1ww*(u2P+u2WWW)/2);
        dF2u2_deta = (9./8) * (F2n*(u2N+u2P)/2 - F2s*(u2P+u2S)/2)
                     - (1./24) * (F2nn*(u2NNN+u2P)/2 - F2ss*(u2P+u2SSS)/2);
        dF3u2_dzeta = (9./8) * (F3u*(u2U+u2P)/2 - F3d*(u2P+u2D)/2)
                      - (1./24) * (F3uu*(u2UUU+u2P)/2 - F3dd*(u2P+u2DDD)/2);

        dF1u3_dxi = (9./8) * (F1e*(u3E+u3P)/2 - F1w*(u3P+u3W)/2)
                    - (1./24) * (F1ee*(u3EEE+u3P)/2 - F1ww*(u3P+u3WWW)/2);
        dF2u3_deta = (9./8) * (F2n*(u3N+u3P)/2 - F2s*(u3P+u3S)/2)
                     - (1./24) * (F2nn*(u3NNN+u3P)/2 - F2ss*(u3P+u3SSS)/2);
        dF3u3_dzeta = (9./8) * (F3u*(u3U+u3P)/2 - F3d*(u3P+u3D)/2)
                      - (1./24) * (F3uu*(u3UUU+u3P)/2 - F3dd*(u3P+u3DDD)/2);

        e(vars->H1, i, j, k) = e(mesh->J, i, j, k)
                               * (dF1u1_dxi + dF2u1_deta + dF3u1_dzeta);
        e(vars->H2, i, j, k) = e(mesh->J, i, j, k)
                               * (dF1u2_dxi + dF2u2_deta + dF3u2_dzeta);
        e(vars->H3, i, j, k) = e(mesh->J, i, j, k)
                               * (dF1u3_dxi + dF2u3_deta + dF3u3_dzeta);
    }
}

void FlowVars_Destroy(FlowVars *vars) {
    // TODO: free all internal arrays.
}

static void Exchg(FlowVars *vars, PartMesh *mesh, int n, ...) {
    va_list ap;
    int cnt;
    double *p[n];

    const int x_exchg_size = 3*n*(mesh->Ny_l+6)*(mesh->Nz_l+6);
    const int y_exchg_size = 3*n*(mesh->Nx_l+6)*(mesh->Nz_l+6);
    const int z_exchg_size = 3*n*(mesh->Nx_l+6)*(mesh->Ny_l+6);

    va_start(ap, n);
    for (int i = 0; i < n; i++) {
        p[i] = va_arg(ap, double *);
    }
    va_end(ap);

    /* X. */
    if (mesh->ri != mesh->Px-1) {
        cnt = 0;
        FOR_INNER_EAST (i, j, k)
            for (int l = 0; l < n; l++)
                vars->x_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->x_exchg, x_exchg_size, MPI_DOUBLE,
                 mesh->nextx, 0, MPI_COMM_WORLD);
    }
    if (mesh->ri != 0) {
        MPI_Recv(vars->x_exchg, x_exchg_size, MPI_DOUBLE,
                 mesh->prevx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_WEST (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->x_exchg[cnt++];
        cnt = 0;
        FOR_INNER_WEST (i, j, k)
            for (int l = 0; l < n; l++)
                vars->x_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->x_exchg, x_exchg_size, MPI_DOUBLE,
                 mesh->prevx, 0, MPI_COMM_WORLD);
    }
    if (mesh->ri != mesh->Px-1) {
        MPI_Recv(vars->x_exchg, x_exchg_size, MPI_DOUBLE,
                 mesh->nextx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_EAST (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->x_exchg[cnt++];
    }

    /* Y. */
    if (mesh->rj != mesh->Py-1) {
        cnt = 0;
        FOR_INNER_NORTH (i, j, k)
            for (int l = 0; l < n; l++)
                vars->y_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->y_exchg, y_exchg_size, MPI_DOUBLE,
                 mesh->nexty, 0, MPI_COMM_WORLD);
    }
    if (mesh->rj != 0) {
        MPI_Recv(vars->y_exchg, y_exchg_size, MPI_DOUBLE,
                 mesh->prevy, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_SOUTH (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->y_exchg[cnt++];
        cnt = 0;
        FOR_INNER_SOUTH (i, j, k)
            for (int l = 0; l < n; l++)
                vars->y_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->y_exchg, y_exchg_size, MPI_DOUBLE,
                 mesh->prevy, 0, MPI_COMM_WORLD);
    }
    if (mesh->rj != mesh->Py-1) {
        MPI_Recv(vars->y_exchg, y_exchg_size, MPI_DOUBLE,
                 mesh->nexty, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_NORTH (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->y_exchg[cnt++];
    }

    /* Z. */
    if (mesh->rk != mesh->Pz-1) {
        cnt = 0;
        FOR_INNER_UP (i, j, k)
            for (int l = 0; l < n; l++)
                vars->z_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->z_exchg, z_exchg_size, MPI_DOUBLE,
                 mesh->nextz, 0, MPI_COMM_WORLD);
    }
    if (mesh->rk != 0) {
        MPI_Recv(vars->z_exchg, z_exchg_size, MPI_DOUBLE,
                 mesh->prevz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_DOWN (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->z_exchg[cnt++];
        cnt = 0;
        FOR_INNER_DOWN (i, j, k)
            for (int l = 0; l < n; l++)
                vars->z_exchg[cnt++] = e(p[l], i, j, k);
        MPI_Send(vars->z_exchg, z_exchg_size, MPI_DOUBLE,
                 mesh->prevz, 0, MPI_COMM_WORLD);
    }
    if (mesh->rk != mesh->Pz-1) {
        MPI_Recv(vars->z_exchg, z_exchg_size, MPI_DOUBLE,
                 mesh->nextz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt = 0;
        FOR_OUTER_UP (i, j, k)
            for (int l = 0; l < n; l++)
                e(p[l], i, j, k) = vars->z_exchg[cnt++];
    }
}