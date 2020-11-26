#ifndef GEO3D_H
#define GEO3D_H

#include <stdio.h>

#define SCALE_INCH_TO_MM 25.4
#define SCALE_MM_TO_INCH (1 / 25.4)

/* 3-dimensional vector. */
typedef struct _vector {
    double x, y, z;         /* Coordinates. */
} Vector;

/* Polyhedron. */
typedef struct _polyhedron Polyhedron;

Vector Vector_add(const Vector, const Vector);
Vector Vector_sub(const Vector, const Vector);
double Vector_dot(const Vector, const Vector);
Vector Vector_crs(const Vector, const Vector);

double Vector_norm(const Vector);
double Vector_dist(const Vector, const Vector);
double Vector_angle(const Vector, const Vector);

Vector Vector_normalize(const Vector);
Vector Vector_lincom(const double, const Vector, const double, const Vector);
Vector Vector_rot(const Vector, const Vector, const double);

Polyhedron *Polyhedron_new(void);
void Polyhedron_destroy(Polyhedron *);
void Polyhedron_read_stl(Polyhedron *, FILE *);
void Polyhedron_print_stats(Polyhedron *);

void Polyhedron_scale(Polyhedron *, double);

void Polyhedron_cpt(
    const Polyhedron *const poly,
    const int nx, const int ny, const int nz,
    const double x[const static nx],
    const double y[const static ny],
    const double z[const static nz],
    double f[const static nx][ny][nz],
    const double maxd
);

#endif