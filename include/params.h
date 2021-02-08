#ifndef IBM3D_PARAMS_H
#define IBM3D_PARAMS_H

typedef struct _params {
    double rho;             /* Density. */
    double mu;              /* Viscosity. */

    double dt;              /* Delta t. */
} Params;

#endif