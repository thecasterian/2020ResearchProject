#include "geo3d.h"

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <glib.h>

static const double PI = 3.141592653589793;

#define max(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a > _b ? _a : _b;})
#define min(a, b) ({typeof(a) _a = a; typeof(b) _b = b; _a < _b ? _a : _b;})

/* Vertex of polyhedron. */
typedef struct _vertex {
    Vector coord;           /* Coordinates. */
    int num_faces;          /* Number of faces containing this vertex. */
    int *face_idx;          /* Indices of faces containing this vertex. */
} Vertex;

/* Edge of polyhedron. */
typedef struct _edge {
    int vertex_idx[2];      /* Indices of two vertices. */
    int face_idx[2];        /* Indices of two faces containing this edge. */
} Edge;

/* Face of polyhedron. */
typedef struct _face {
    int vertex_idx[3];      /* Indices of three vertices. */
    int edge_idx[3];        /* Indices of three edges. */
    Vector n;               /* Outward unit normal vector. */
} Face;

struct _polyhedron {
    size_t num_vertices;    /* Number of vertices. */
    size_t num_edges;       /* Number of edges. */
    size_t num_faces;       /* Number of faces. */

    Vertex *vertex_list;    /* List of vertices. */
    Edge *edge_list;        /* List of edges. */
    Face *face_list;        /* List of faces. */
};

typedef struct _face_extrusion {
    Vector pts[6];
    int sgn;
} FaceExtrusion;

typedef struct _edge_extrusion {
    Vector pts[6];
    int sgn;
} EdgeExtrusion;

typedef struct _vertex_extrusion {
    Vector vertex;
    Vector axis;
    double radius;
    double angle;
    int sgn;
} VertexExtrusion;

typedef struct _plane {
    union {
        struct {
            double a, b, c;
        };
        Vector n;
    };
    double d;
} Plane;

typedef struct _pair_int {
    int first, second;
} PairInt;

typedef struct _vert_tree_value {
    int vert_idx, num_faces, face_idx;
} VertTreeValue;

typedef struct _edge_tree_value {
    int edge_idx, face_idx[2];
} EdgeTreeValue;

static int cmp_vertex(const void *, const void *, void *);
static int cmp_pairint(const void *, const void *, void *);
static int store_vertex(void *, void *, void *);
static int store_edge(void *, void *, void *);

static inline Plane Plane_3pts(Vector, Vector, Vector);
static inline double Plane_dist(Plane, Vector);

static inline double Segment_dist(Vector, Vector, Vector);

static inline void VertexExtrusion_cpt(
    const VertexExtrusion,
    const int, const int, const int,
    const double [], const double [], const double [],
    double [][*][*]
);
static inline void EdgeExtrusion_cpt(
    const EdgeExtrusion,
    const int, const int, const int,
    const double [], const double [], const double [],
    double [][*][*]
);
static inline void FaceExtrusion_cpt(
    const FaceExtrusion,
    const int, const int, const int,
    const double [], const double [], const double [],
    double [][*][*]
);

Vector Vector_add(const Vector a, const Vector b) {
    Vector res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    return res;
}

Vector Vector_sub(const Vector a, const Vector b) {
    Vector res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    return res;
}

double Vector_dot(const Vector a, const Vector b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vector Vector_crs(const Vector a, const Vector b) {
    Vector res;
    res.x = a.y*b.z - a.z*b.y;
    res.y = a.z*b.x - a.x*b.z;
    res.z = a.x*b.y - a.y*b.x;
    return res;
}

double Vector_norm(const Vector v) {
    return sqrt(Vector_dot(v, v));
}

double Vector_dist(const Vector a, const Vector b) {
    return Vector_norm(Vector_sub(a, b));
}

double Vector_angle(const Vector a, const Vector b) {
    double lena = Vector_norm(a);
    double lenb = Vector_norm(b);

    double costheta = Vector_dot(a, b) / (lena*lenb);
    if (costheta < -1)
        costheta = -1;
    else if (costheta > 1)
        costheta = 1;

    return acos(costheta);
}

Vector Vector_normalize(const Vector v) {
    Vector res = v;
    const double len = Vector_norm(v);
    res.x = v.x / len;
    res.y = v.y / len;
    res.z = v.z / len;
    return res;
}

Vector Vector_lincom(const double a, const Vector u, const double b, const Vector v) {
    Vector res;
    res.x = a*u.x + b*v.x;
    res.y = a*u.y + b*v.y;
    res.z = a*u.z + b*v.z;
    return res;
}

Vector Vector_rot(const Vector v, const Vector axis, const double angle) {
    const Vector n = Vector_normalize(axis);
    Vector res;
    res.x = (cos(angle)+n.x*n.x*(1-cos(angle))) * v.x
            + (n.x*n.y*(1-cos(angle))-n.z*sin(angle)) * v.y
            + (n.x*n.z*(1-cos(angle))+n.y*sin(angle)) * v.z;
    res.y = (n.x*n.y*(1-cos(angle))+n.z*sin(angle)) * v.x
            + (cos(angle)+n.y*n.y*(1-cos(angle))) * v.y
            + (n.y*n.z*(1-cos(angle))-n.x*sin(angle)) * v.z;
    res.z = (n.x*n.z*(1-cos(angle))-n.y*sin(angle)) * v.x
            + (n.y*n.z*(1-cos(angle))+n.x*sin(angle)) * v.y
            + (cos(angle)+n.z*n.z*(1-cos(angle))) * v.z;
    return res;
}

/* Allocate new polyhedron. */
Polyhedron *Polyhedron_new(void) {
    Polyhedron *poly = malloc(sizeof(Polyhedron));
    poly->num_vertices = poly->num_edges = poly->num_faces = 0;
    poly->vertex_list = NULL;
    poly->edge_list = NULL;
    poly->face_list = NULL;
    return poly;
}

void Polyhedron_destroy(Polyhedron *poly) {
    for (int i = 0; i < poly->num_vertices; i++) {
        free(poly->vertex_list[i].face_idx);
    }
    free(poly->vertex_list);
    free(poly->edge_list);
    free(poly->face_list);
    free(poly);
}

/* Read (binary) STL file. */
void Polyhedron_read_stl(Polyhedron *poly, FILE *f) {
    char header[81] = "";
    uint32_t num_faces;
    float coord[3];
    Vector vtmp[3];
    uint16_t attr;

    /* Read header (first 80 bytes). */
    fread(header, 1, 80, f);

    /* Read number of faces. (next 4-bytes unsigned integer) */
    fread(&num_faces, 4, 1, f);
    poly->num_faces = num_faces;

    /* Allocate face list. */
    poly->face_list = calloc(poly->num_faces, sizeof(Face));

    /* Vertex search tree. Vector -> VertTreeValue */
    GTree *vert_tree = g_tree_new_full(cmp_vertex, NULL, g_free, g_free);

    for (int i = 0; i < poly->num_faces; i++) {
        /* Read normal vector. (3 x 4 bytes) */
        fread(coord, 4, 3, f);
        poly->face_list[i].n.x = coord[0];
        poly->face_list[i].n.y = coord[1];
        poly->face_list[i].n.z = coord[2];
        /* Read coordinates of 3 vertices temporarily in `vtmp`.
           (3 x 3 x 4 bytes) */
        for (int j = 0; j < 3; j++) {
            fread(coord, 4, 3, f);
            vtmp[j].x = coord[0];
            vtmp[j].y = coord[1];
            vtmp[j].z = coord[2];
        }
        /* Read attribute. (2 bytes; discarded) */
        fread(&attr, 2, 1, f);

        for (int j = 0; j < 3; j++) {
            /* Check if vertex is already indexed. */
            VertTreeValue *v = g_tree_lookup(vert_tree, &vtmp[j]);

            if (!v) {
                /* If not indexed, insert it to vertex search tree. */
                Vector *key = malloc(sizeof(Vector));
                VertTreeValue *value = malloc(sizeof(VertTreeValue));

                memcpy(key, &vtmp[j], sizeof(Vector));
                value->vert_idx = g_tree_nnodes(vert_tree);
                value->num_faces = 0;
                value->face_idx = i;

                g_tree_insert(vert_tree, key, value);

                v = value;
            }

            /* Store the index of vertex in triangle */
            poly->face_list[i].vertex_idx[j] = v->vert_idx;

            v->num_faces++;
        }
    }

    /* Allocate vertex list and store the contents of vertex search tree into
       the list. */
    poly->num_vertices = g_tree_nnodes(vert_tree);
    poly->vertex_list = calloc(poly->num_vertices, sizeof(Vertex));
    g_tree_foreach(vert_tree, store_vertex, poly->vertex_list);

    g_tree_destroy(vert_tree);

    /* Edge search tree. PairInt -> EdgeTreeValue */
    GTree *edge_tree = g_tree_new_full(cmp_pairint, NULL, g_free, g_free);

    PairInt etmp;

    for (int i = 0; i < poly->num_faces; i++) {
        for (int j = 0; j < 3; j++) {
            etmp.first = poly->face_list[i].vertex_idx[j];
            etmp.second = poly->face_list[i].vertex_idx[(j+1)%3];

            /* Check if vertex is already indexed. */
            EdgeTreeValue *v = g_tree_lookup(edge_tree, &etmp);

            if (!v) {
                /* If not indexed, insert it to edge search tree. */
                PairInt *key = malloc(sizeof(PairInt));
                EdgeTreeValue *value = malloc(sizeof(EdgeTreeValue));

                memcpy(key, &etmp, sizeof(PairInt));
                value->edge_idx = g_tree_nnodes(edge_tree);
                value->face_idx[0] = i;

                g_tree_insert(edge_tree, key, value);

                v = value;
            }
            else {
                /* If indexed, set one remained face index. */
                v->face_idx[1] = i;
            }

            /* Set edge index of face. */
            poly->face_list[i].edge_idx[j] = v->edge_idx;
        }
    }

    /* Allocate edge list and store the contents of egde search tree into the
       list. */
    poly->num_edges = g_tree_nnodes(edge_tree);
    poly->edge_list = calloc(poly->num_edges, sizeof(Edge));
    g_tree_foreach(edge_tree, store_edge, poly->edge_list);

    g_tree_destroy(edge_tree);

    /* For each vertex, find faces containing it. */
    for (int i = 0; i < poly->num_vertices; i++) {
        int cur_face_idx = poly->vertex_list[i].face_idx[0];
        int cur_edge_idx;
        for (int j = 1; j < poly->vertex_list[i].num_faces; j++) {
            if (poly->face_list[cur_face_idx].vertex_idx[0] == i) {
                cur_edge_idx = poly->face_list[cur_face_idx].edge_idx[2];
            }
            else if (poly->face_list[cur_face_idx].vertex_idx[1] == i) {
                cur_edge_idx = poly->face_list[cur_face_idx].edge_idx[0];
            }
            else {
                cur_edge_idx = poly->face_list[cur_face_idx].edge_idx[1];
            }

            if (poly->edge_list[cur_edge_idx].face_idx[0] == cur_face_idx) {
                cur_face_idx = poly->edge_list[cur_edge_idx].face_idx[1];
            }
            else {
                cur_face_idx = poly->edge_list[cur_edge_idx].face_idx[0];
            }

            poly->vertex_list[i].face_idx[j] = cur_face_idx;
        }
    }

    /* DEBUG */
#ifdef DEBUG
    for (int i = 0; i < poly->num_vertices; i++) {
        printf("vertex %d\n", i);
        printf("  faces:");
        for (int j = 0; j < poly->vertex_list[i].num_faces; j++) {
            printf(" %d", poly->vertex_list[i].face_idx[j]);
        }
        printf("\n");
    }
    for (int i = 0; i < poly->num_edges; i++) {
        printf("edge %d\n", i);
        printf("  vertices: %d %d\n",
               poly->edge_list[i].vertex_idx[0],
               poly->edge_list[i].vertex_idx[1]);
        printf("  faces: %d %d\n",
               poly->edge_list[i].face_idx[0],
               poly->edge_list[i].face_idx[1]);
    }
    for (int i = 0; i < poly->num_faces; i++) {
        printf("face %d\n", i);
        printf("  vertices: %d %d %d\n",
               poly->face_list[i].vertex_idx[0],
               poly->face_list[i].vertex_idx[1],
               poly->face_list[i].vertex_idx[2]);
        printf("  edges: %d %d %d\n",
               poly->face_list[i].edge_idx[0],
               poly->face_list[i].edge_idx[1],
               poly->face_list[i].edge_idx[2]);
    }
#endif
}

void Polyhedron_print_stats(Polyhedron *poly) {
    double xmin = INFINITY, xmax = -INFINITY;
    double ymin = INFINITY, ymax = -INFINITY;
    double zmin = INFINITY, zmax = -INFINITY;

    printf(
        "Input polyhedron size: %zu faces, %zu edges, %zu vertices\n",
        poly->num_faces,
        poly->num_edges,
        poly->num_vertices
    );
    for (int i = 0; i < poly->num_vertices; i++) {
        const Vector v = poly->vertex_list[i].coord;
        xmin = min(xmin, v.x);
        xmax = max(xmax, v.x);
        ymin = min(ymin, v.y);
        ymax = max(ymax, v.y);
        zmin = min(zmin, v.z);
        zmax = max(zmax, v.z);
    }
    printf("  xmin: %10.4lf, xmax: %10.4lf\n", xmin, xmax);
    printf("  ymin: %10.4lf, ymax: %10.4lf\n", ymin, ymax);
    printf("  zmin: %10.4lf, zmax: %10.4lf\n", zmin, zmax);
}

void Polyhedron_cpt(
    const Polyhedron *const poly,
    const int nx, const int ny, const int nz,
    const double x[const restrict static nx],
    const double y[const restrict static ny],
    const double z[const restrict static nz],
    double f[const restrict static nx][ny][nz],
    const double maxd
) {
    VertexExtrusion ve;
    EdgeExtrusion ee;
    FaceExtrusion fe;

    /* Initialize distance function array. */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                f[i][j][k] = NAN;
            }
        }
    }

    /* Face extrusions. */
    for (int i = 0; i < poly->num_faces; i++) {
        /* Outward extrusions. */
        for (int j = 0; j < 3; j++) {
            fe.pts[j] = poly->vertex_list[poly->face_list[i].vertex_idx[j]].coord;
            fe.pts[j+3] = Vector_lincom(1, fe.pts[j], maxd, poly->face_list[i].n);
        }
        fe.sgn = 1;
        FaceExtrusion_cpt(fe, nx, ny, nz, x, y, z, f);

        /* Inward extrusions. */
        for (int j = 0; j < 3; j++) {
            fe.pts[j] = poly->vertex_list[poly->face_list[i].vertex_idx[(3-j)%3]].coord;
            fe.pts[j+3] = Vector_lincom(1, fe.pts[j], -maxd, poly->face_list[i].n);
        }
        fe.sgn = -1;
        FaceExtrusion_cpt(fe, nx, ny, nz, x, y, z, f);
    }

    /* Edge extrusions. */
    for (int i = 0; i < poly->num_edges; i++) {
        ee.pts[0] = poly->vertex_list[poly->edge_list[i].vertex_idx[0]].coord;
        ee.pts[1] = poly->vertex_list[poly->edge_list[i].vertex_idx[1]].coord;

        const Vector v = Vector_sub(ee.pts[1], ee.pts[0]);
        const Vector n_left = poly->face_list[poly->edge_list[i].face_idx[0]].n;
        const Vector n_right = poly->face_list[poly->edge_list[i].face_idx[1]].n;
        ee.sgn = Vector_dot(Vector_crs(v, n_left), n_right) >= 0 ? 1 : -1;

        double angle = Vector_angle(n_left, n_right);
        if (angle <= PI/3) {
            if (ee.sgn == 1) {
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n_left);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n_left);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n_right);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n_right);
            }
            else {
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n_right);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n_right);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n_left);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n_left);
            }
            EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
        }
        else if (angle <= 2*PI/3) {
            const Vector n_mid = Vector_rot(n_left, v, angle/2);
            if (ee.sgn == 1) {
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n_left);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n_left);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n_mid);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n_mid);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n_mid);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n_mid);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n_right);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n_right);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
            }
            else {
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n_right);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n_right);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n_mid);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n_mid);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n_mid);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n_mid);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n_left);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n_left);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
            }
        }
        else {
            if (ee.sgn == 1) {
                const Vector n1 = Vector_rot(n_left, v, angle/3);
                const Vector n2 = Vector_rot(n_left, v, 2*angle/3);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n_left);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n_left);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n1);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n1);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n1);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n1);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n2);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n2);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], maxd, n2);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], maxd, n2);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], maxd, n_right);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], maxd, n_right);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
            }
            else {
                const Vector n1 = Vector_rot(n_left, v, -angle/3);
                const Vector n2 = Vector_rot(n_left, v, -2*angle/3);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n_right);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n_right);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n2);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n2);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n2);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n2);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n1);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n1);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
                ee.pts[2] = Vector_lincom(1, ee.pts[0], -maxd, n1);
                ee.pts[3] = Vector_lincom(1, ee.pts[1], -maxd, n1);
                ee.pts[4] = Vector_lincom(1, ee.pts[0], -maxd, n_left);
                ee.pts[5] = Vector_lincom(1, ee.pts[1], -maxd, n_left);
                EdgeExtrusion_cpt(ee, nx, ny, nz, x, y, z, f);
            }
        }
    }

    /* Vertex extrusions. */
    for (int i = 0; i < poly->num_vertices; i++) {
        int adj_vertex_idx[poly->vertex_list[i].num_faces];
        double adj_angle = 0;
        bool is_convex = true, is_concave = true;

        ve.vertex = poly->vertex_list[i].coord;
        ve.radius = maxd;

        /* Calculate axis (pseudonormal). */
        ve.axis = (Vector){0, 0, 0};
        for (int j = 0; j < poly->vertex_list[i].num_faces; j++) {
            const Face adj_face = poly->face_list[poly->vertex_list[i].face_idx[j]];
            for (int k = 0; k < 3; k++) {
                if (adj_face.vertex_idx[k] == i) {
                    adj_angle = Vector_angle(
                        Vector_sub(
                            poly->vertex_list[adj_face.vertex_idx[(k+1)%3]].coord,
                            ve.vertex
                        ),
                        Vector_sub(
                            poly->vertex_list[adj_face.vertex_idx[(k+2)%3]].coord,
                            ve.vertex
                        )
                    );
                    adj_vertex_idx[j] = adj_face.vertex_idx[(k+2)%3];
                }
            }
            ve.axis = Vector_lincom(1, ve.axis, adj_angle, adj_face.n);
        }
        ve.axis = Vector_normalize(ve.axis);

        /* Calculate cone angle. */
        ve.angle = 0;
        for (int j = 0; j < poly->vertex_list[i].num_faces; j++) {
            const Vector adj_normal = poly->face_list[poly->vertex_list[i].face_idx[j]].n;
            if (Vector_dot(ve.axis, adj_normal) > 0) {
                ve.angle = max(ve.angle, Vector_angle(ve.axis, adj_normal));
            }
        }

        /* Calculate convexity. */
        for (int j = 0; j < poly->vertex_list[i].num_faces; j++) {
            double d = Vector_dot(
                Vector_sub(
                    poly->vertex_list[adj_vertex_idx[j]].coord,
                    ve.vertex
                ),
                ve.axis
            );
            if (d > 0) {
                is_convex = false;
            }
            if (d < 0) {
                is_concave = false;
            }
        }

        if (!is_concave) {
            ve.sgn = 1;
            VertexExtrusion_cpt(ve, nx, ny, nz, x, y, z, f);
        }
        if (!is_convex) {
            ve.sgn = -1;
            ve.axis.x = -ve.axis.x;
            ve.axis.y = -ve.axis.y;
            ve.axis.z = -ve.axis.z;
            VertexExtrusion_cpt(ve, nx, ny, nz, x, y, z, f);
        }
    }

    /* Fill unset points. */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 1; k < nz; k++) {
                if (isnan(f[i][j][k]) && f[i][j][k-1] < 0) {
                    for (; isnan(f[i][j][k]); k++) {
                        f[i][j][k] = -maxd;
                    }
                }
            }
        }
    }
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                if (isnan(f[i][j][k])) {
                    f[i][j][k] = maxd;
                }
            }
        }
    }
}

static inline Plane Plane_3pts(Vector a, Vector b, Vector c) {
    Vector ab = Vector_sub(b, a);
    Vector ac = Vector_sub(c, a);
    Vector n = Vector_crs(ab, ac);

    Plane res;
    res.n = n;
    res.d = -Vector_dot(n, a);

    return res;
}

static inline double Plane_dist(Plane p, Vector v) {
    return fabs(p.a*v.x+p.b*v.y+p.c*v.z+p.d) / Vector_norm(p.n);
}

static inline double Segment_dist(Vector a, Vector b, Vector v) {
    const Vector ab = Vector_sub(b, a);
    const Vector av = Vector_sub(v, a);
    const double t = Vector_dot(ab, av) / Vector_dot(ab, ab);
    const Vector h = Vector_lincom(1-t, a, t, b);
    return Vector_dist(v, h);
}

static int cmp_vertex(const void *_v1, const void *_v2, void *_aux G_GNUC_UNUSED) {
    const Vertex *v1 = _v1, *v2 = _v2;
    if (v1->coord.x != v2->coord.x)
        return v1->coord.x < v2->coord.x ? -1 : 1;
    if (v1->coord.y != v2->coord.y)
        return v1->coord.y < v2->coord.y ? -1 : 1;
    if (v1->coord.z != v2->coord.z)
        return v1->coord.z < v2->coord.z ? -1 : 1;
    return 0;
}

static int store_vertex(void *_key, void *_value, void *_aux) {
    Vector *key = _key;
    VertTreeValue *value = _value;
    Vertex *vertex_list = _aux;

    /* Store coordinates. */
    memcpy(&vertex_list[value->vert_idx].coord, key, sizeof(Vector));
    /* Store number of faces. */
    vertex_list[value->vert_idx].num_faces = value->num_faces;
    /* Allocate face list of vertex. */
    vertex_list[value->vert_idx].face_idx = malloc(value->num_faces * sizeof(int));
    /* Store index of a face. */
    vertex_list[value->vert_idx].face_idx[0] = value->face_idx;
    for (int i = 1; i < value->num_faces; i++) {
        vertex_list[value->vert_idx].face_idx[i] = -1;
    }

    return 0;
}

static int store_edge(void *_key, void *_value, void *_aux) {
    PairInt *key = _key;
    EdgeTreeValue *value = _value;
    Edge *edge_list = _aux;

    /* Store indices of verticies. */
    memcpy(edge_list[value->edge_idx].vertex_idx, key, sizeof(PairInt));
    /* Store indices of faces. */
    memcpy(edge_list[value->edge_idx].face_idx, value->face_idx, 2*sizeof(int));

    return 0;
}

static int cmp_pairint(const void *_v1, const void *_v2, void *_aux G_GNUC_UNUSED) {
    const PairInt *v1 = _v1, *v2 = _v2;
    PairInt u1 = {min(v1->first, v1->second), max(v1->first, v1->second)};
    PairInt u2 = {min(v2->first, v2->second), max(v2->first, v2->second)};

    if (u1.first != u2.first)
        return u1.first - u2.first;
    return u1.second - u2.second;
}

static inline void VertexExtrusion_cpt(
    const VertexExtrusion ve,
    const int nx, const int ny, const int nz,
    const double x[const restrict static nx],
    const double y[const restrict static ny],
    const double z[const restrict static nz],
    double f[const restrict static nx][ny][nz]
) {
    int xmax_idx, xmin_idx, ymax_idx, ymin_idx, zmax_idx, zmin_idx;

    Vector d;
    double dist;

    xmax_idx = upper_bound(nx, x, ve.vertex.x + ve.radius);
    xmin_idx = lower_bound(nx, x, ve.vertex.x - ve.radius);
    ymax_idx = upper_bound(ny, y, ve.vertex.y + ve.radius);
    ymin_idx = lower_bound(ny, y, ve.vertex.y - ve.radius);
    zmax_idx = upper_bound(nz, z, ve.vertex.z + ve.radius);
    zmin_idx = lower_bound(nz, z, ve.vertex.z - ve.radius);

    for (int i = xmin_idx; i < xmax_idx; i++) {
        for (int j = ymin_idx; j < ymax_idx; j++) {
            for (int k = zmin_idx; k < zmax_idx; k++) {
                d = Vector_sub((Vector){x[i], y[j], z[k]}, ve.vertex);
                dist = Vector_norm(d);
                if (Vector_angle(ve.axis, d) <= ve.angle + 1e-12 && dist <= ve.radius + 1e-12) {
                    dist = Vector_norm(d);
                    if (isnan(f[i][j][k]) || dist < fabs(f[i][j][k])) {
                        f[i][j][k] = ve.sgn * dist;
                    }
                }
            }
        }
    }
}

static inline void EdgeExtrusion_cpt(
    const EdgeExtrusion ee,
    const int nx, const int ny, const int nz,
    const double x[const restrict static nx],
    const double y[const restrict static ny],
    const double z[const restrict static nz],
    double f[const restrict static nx][ny][nz]
) {
    double xmax = -INFINITY, xmin = INFINITY;
    double ymax = -INFINITY, ymin = INFINITY;
    double zmax = -INFINITY, zmin = INFINITY;
    int xmax_idx, xmin_idx, ymax_idx, ymin_idx, zmax_idx, zmin_idx;

    Plane ee_faces[5];
    bool is_in;
    double dist;

    for (int i = 0; i < 6; i++) {
        xmax = max(xmax, ee.pts[i].x);
        xmin = min(xmin, ee.pts[i].x);
        ymax = max(ymax, ee.pts[i].y);
        ymin = min(ymin, ee.pts[i].y);
        zmax = max(zmax, ee.pts[i].z);
        zmin = min(zmin, ee.pts[i].z);
    }

    xmax_idx = upper_bound(nx, x, xmax);
    xmin_idx = lower_bound(nx, x, xmin);
    ymax_idx = upper_bound(ny, y, ymax);
    ymin_idx = lower_bound(ny, y, ymin);
    zmax_idx = upper_bound(nz, z, zmax);
    zmin_idx = lower_bound(nz, z, zmin);

    ee_faces[0] = Plane_3pts(ee.pts[0], ee.pts[4], ee.pts[2]);
    ee_faces[1] = Plane_3pts(ee.pts[1], ee.pts[3], ee.pts[5]);
    ee_faces[2] = Plane_3pts(ee.pts[0], ee.pts[2], ee.pts[3]);
    ee_faces[3] = Plane_3pts(ee.pts[2], ee.pts[4], ee.pts[5]);
    ee_faces[4] = Plane_3pts(ee.pts[0], ee.pts[1], ee.pts[5]);

    for (int i = xmin_idx; i < xmax_idx; i++) {
        for (int j = ymin_idx; j < ymax_idx; j++) {
            for (int k = zmin_idx; k < zmax_idx; k++) {
                const Vector cur = {x[i], y[j], z[k]};
                is_in = true;
                for (int l = 0; l < 5; l++) {
                    if (Vector_dot(ee_faces[l].n, cur) + ee_faces[l].d > 1e-12) {
                        is_in = false;
                    }
                }
                if (is_in) {
                    dist = Segment_dist(ee.pts[0], ee.pts[1], cur);
                    if (isnan(f[i][j][k]) || dist < fabs(f[i][j][k])) {
                        f[i][j][k] = ee.sgn * dist;
                    }
                }
            }
        }
    }
}

static inline void FaceExtrusion_cpt(
    const FaceExtrusion fe,
    const int nx, const int ny, const int nz,
    const double x[const restrict static nx],
    const double y[const restrict static ny],
    const double z[const restrict static nz],
    double f[const restrict static nx][ny][nz]
) {
    double xmax = -INFINITY, xmin = INFINITY;
    double ymax = -INFINITY, ymin = INFINITY;
    double zmax = -INFINITY, zmin = INFINITY;
    int xmax_idx, xmin_idx, ymax_idx, ymin_idx, zmax_idx, zmin_idx;

    Plane fe_faces[5];
    bool is_in;
    double dist;

    for (int i = 0; i < 6; i++) {
        xmax = max(xmax, fe.pts[i].x);
        xmin = min(xmin, fe.pts[i].x);
        ymax = max(ymax, fe.pts[i].y);
        ymin = min(ymin, fe.pts[i].y);
        zmax = max(zmax, fe.pts[i].z);
        zmin = min(zmin, fe.pts[i].z);
    }

    xmax_idx = upper_bound(nx, x, xmax);
    xmin_idx = lower_bound(nx, x, xmin);
    ymax_idx = upper_bound(ny, y, ymax);
    ymin_idx = lower_bound(ny, y, ymin);
    zmax_idx = upper_bound(nz, z, zmax);
    zmin_idx = lower_bound(nz, z, zmin);

    fe_faces[0] = Plane_3pts(fe.pts[0], fe.pts[2], fe.pts[1]);
    fe_faces[1] = Plane_3pts(fe.pts[3], fe.pts[4], fe.pts[5]);
    fe_faces[2] = Plane_3pts(fe.pts[0], fe.pts[1], fe.pts[4]);
    fe_faces[3] = Plane_3pts(fe.pts[1], fe.pts[2], fe.pts[5]);
    fe_faces[4] = Plane_3pts(fe.pts[0], fe.pts[3], fe.pts[5]);

    for (int i = xmin_idx; i < xmax_idx; i++) {
        for (int j = ymin_idx; j < ymax_idx; j++) {
            for (int k = zmin_idx; k < zmax_idx; k++) {
                const Vector cur = {x[i], y[j], z[k]};
                is_in = true;
                for (int l = 0; l < 5; l++) {
                    if (Vector_dot(fe_faces[l].n, cur) + fe_faces[l].d > 1e-12) {
                        is_in = false;
                    }
                }
                if (is_in) {
                    dist = Plane_dist(fe_faces[0], cur);
                    if (isnan(f[i][j][k]) || dist < fabs(f[i][j][k])) {
                        f[i][j][k] = fe.sgn * dist;
                    }
                }
            }
        }
    }
}

/* Find the index of the first element in ARR which is greater than or equal
   to VAL. ARR must be sorted in increasing order. */
int lower_bound(const int len, const double arr[const static len], const double val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val <= arr[mid]) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

/* Find the index of the first element in ARR which is greater than VAL. ARR
   must be sorted in increasing order. */
int upper_bound(const int len, const double arr[const static len], const double val) {
    int l = 0;
    int h = len;
    while (l < h) {
        int mid =  l + (h - l) / 2;
        if (val >= arr[mid]) {
            l = mid + 1;
        }
        else {
            h = mid;
        }
    }
    return l;
}
