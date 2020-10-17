#include "geo3d.h"

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <glib.h>

const double PI = 3.141592653589793;

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
    double height;
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
static double vecang(Vector, Vector, Vector);

static int int_min(int, int);
static int int_max(int, int);

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

/* Allocate new polyhedron. */
Polyhedron *Polyhedron_new(void) {
    Polyhedron *poly = malloc(sizeof(Polyhedron));
    poly->num_vertices = poly->num_edges = poly->num_faces = 0;
    poly->vertex_list = NULL;
    poly->edge_list = NULL;
    poly->face_list = NULL;
    return poly;
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
}

void Polyhedron_cpt(
    const Polyhedron *const poly,
    const int nx, const int ny, const int nz,
    const double x[const static nx],
    const double y[const static ny],
    const double z[const static nz],
    double f[const static nx][ny][nz]
) {

}

Plane Plane_3pts(Vector a, Vector b, Vector c) {
    Vector ab = Vector_sub(b, a);
    Vector ac = Vector_sub(c, a);
    Vector n = Vector_crs(ab, ac);

    Plane res;
    res.n = n;
    res.d = -Vector_dot(n, a);

    return res;
}

Vector Plane_proj(Plane p, Vector v) {
    double k = (p.a*v.x+p.b*v.y+p.c*v.z+p.d) / (p.a*p.a+p.b*p.b+p.c*p.c);
    Vector res;
    res.x = v.x - k*p.a;
    res.y = v.y - k*p.b;
    res.z = v.z - k*p.c;
    return res;
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
    PairInt u1 = {int_min(v1->first, v1->second), int_max(v1->first, v1->second)};
    PairInt u2 = {int_min(v2->first, v2->second), int_max(v2->first, v2->second)};

    if (u1.first != u2.first)
        return u1.first - u2.first;
    return u1.second - u2.second;
}

static double vecang(Vector a, Vector b, Vector axis) {
    double lena = Vector_norm(a);
    double lenb = Vector_norm(b);

    double costheta = Vector_dot(a, b) / (lena*lenb);
    if (costheta < -1)
        costheta = -1;
    else if (costheta > 1)
        costheta = 1;

    double theta = acos(costheta);
    Vector crsprd = Vector_crs(a, b);
    double projcoeff = Vector_dot(crsprd, axis) / Vector_dot(axis, axis);
    if (projcoeff < 0)
        theta = 2*acos(-1) - theta;

    return theta;
}

static int int_min(int a, int b) {
    return a < b ? a : b;
}

static int int_max(int a, int b) {
    return a > b ? a : b;
}
