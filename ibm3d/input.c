#include "ibm3d.h"

#include <stdio.h>
#include <stdlib.h>

struct _input_params {
    /* Name of stl file. */
    char stl_file[100];

    /* Number of cells in x, y, and z direction, respectively. */
    int Nx, Ny, Nz;
    /* Coordinates of cell faces. */
    double *xf, *yf, *zf;
    /* Reynolds number. */
    double Re;
    /* Delta t. */
    double dt;
    /* Totla number of time steps. */
    int num_time_steps;

    /* Initialize velocities and pressure from file? (T/F) */
    int init_using_file;
    /* Names of velocity input files for initialization. */
    char init_file_u1[100], init_file_u2[100], init_file_u3[100];
    /* Names of pressure input file for initialization. */
    char init_file_p[100];

    /* Name of velocity output files for result export */
    char output_file_u1[100], output_file_u2[100], output_file_u3[100];
    /* Name of pressure output file for result export */
    char output_file_p[100];
};

InputParams *InputParams_new(void) {
    InputParams *res = calloc(1, sizeof(InputParams));
    res->xf = res->yf = res->zf = NULL;
    return res;
}

void InputParams_read_file(InputParams *params, const char *filename) {
    /* Open input file. */
    FILE *fp = fopen(filename, "r");

    /* Read stl file name. */
    fscanf(fp, "%*s %s", params->stl_file);

    /* Read grid geometry. */
    fscanf(fp, "%*s %d", &params->Nx);
    params->xf = calloc(params->Nx+1, sizeof(double));
    for (int i = 0; i <= params->Nx; i++) {
        fscanf(fp, "%lf", &params->xf[i]);
    }

    fscanf(fp, "%*s %d", &params->Ny);
    params->yf = calloc(params->Ny+1, sizeof(double));
    for (int j = 0; j <= params->Ny; j++) {
        fscanf(fp, "%lf", &params->yf[j]);
    }

    fscanf(fp, "%*s %d", &params->Nz);
    params->zf = calloc(params->Nz+1, sizeof(double));
    for (int k = 0; k <= params->Nz; k++) {
        fscanf(fp, "%lf", &params->zf[k]);
    }

    /* Read Reynolds number, delta t, and number of time steps. */
    fscanf(fp, "%*s %lf", &params->Re);
    fscanf(fp, "%*s %lf %*s %d", &params->dt, &params->num_time_steps);

    /* Read initialization file names. */
    fscanf(fp, "%*s %d", &params->init_using_file);
    fscanf(fp, "%*s %s", params->init_file_u1);
    fscanf(fp, "%*s %s", params->init_file_u2);
    fscanf(fp, "%*s %s", params->init_file_u3);
    fscanf(fp, "%*s %s", params->init_file_p);

    /* Read output file names */
    fscanf(fp, "%*s %s", params->output_file_u1);
    fscanf(fp, "%*s %s", params->output_file_u2);
    fscanf(fp, "%*s %s", params->output_file_u3);
    fscanf(fp, "%*s %s", params->output_file_p);

    fclose(fp);
}

void InputParams_destroy(InputParams *params) {
    free(params->xf);
    free(params->yf);
    free(params->zf);
}
