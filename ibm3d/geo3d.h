#ifndef GEO3D_H
#define GEO3D_H

#include <stdio.h>

typedef struct _geo3dvector {
    double x, y, z;
} Geo3dVector;

typedef struct _geo3dpolyhedronface {
    int vertex_idx[3];
    Geo3dVector n;

    int adj_face_idx[3];
    double adj_face_angle[3];
} Geo3dPolyhedronFace;

typedef struct _geo3dpolyhedron {
    size_t num_vertices, num_faces;
    Geo3dVector *vertex_list;
    Geo3dPolyhedronFace *face_list;
} Geo3dPolyhedron;

typedef struct _geo3dplane {
    union {
        struct {
            double a, b, c;
        };
        Geo3dVector n;
    };
    double d;
} Geo3dPlane;

Geo3dVector Geo3dVector_add(const Geo3dVector, const Geo3dVector);
Geo3dVector Geo3dVector_sub(const Geo3dVector, const Geo3dVector);
double      Geo3dVector_dot(const Geo3dVector, const Geo3dVector);
Geo3dVector Geo3dVector_crs(const Geo3dVector, const Geo3dVector);

double Geo3dVector_norm(const Geo3dVector);
double Geo3dVector_dist(const Geo3dVector, const Geo3dVector);

void Geo3dPolyhedron_init(Geo3dPolyhedron *);
void Geo3dPolyhedron_read_stl(Geo3dPolyhedron *, FILE *);

double Geo3dPolyhedron_sgndist(Geo3dPolyhedron *, Geo3dVector);

Geo3dPlane Geo3dPlane_3pts(Geo3dVector, Geo3dVector, Geo3dVector);
Geo3dVector Geo3dPlane_proj(Geo3dPlane, Geo3dVector);

#endif