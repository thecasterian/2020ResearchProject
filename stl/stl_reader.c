#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <glib.h>

#define SWAP(a, b) ({ __auto_type tmp = a; a = b; b = tmp;})

struct stl_vector3 {
    union {
        struct {
            float x, y, z;
        };
        float c[3];
    };
};

struct stl_triangle {
    int vertidx[3];
    struct stl_vector3 n;
    int adjtriidx[3];
    float dihed[3];
};

struct pairint {
    int first, second;
};

struct stl {
    size_t nvert, ntri;
    struct stl_vector3 *verts;
    struct stl_triangle *tris;
};

int cmp_vertex(const void *v1, const void *v2, void *aux G_GNUC_UNUSED) {
    const struct stl_vector3 *vt1 = v1, *vt2 = v2;
    if (vt1->x != vt2->x)
        return vt1->x < vt2->x ? -1 : 1;
    if (vt1->y != vt2->y)
        return vt1->y < vt2->y ? -1 : 1;
    if (vt1->z != vt2->z)
        return vt1->z < vt2->z ? -1 : 1;
    return 0;
}

int store_vertex(void *key, void *value, void *aux) {
    struct stl_vector3 *verts = aux;
    int idx = *(int *)value;
    memcpy(&verts[idx], key, sizeof(struct stl_vector3));
    return 0;
}

int cmp_pairint(const void *v1, const void *v2, void *aux G_GNUC_UNUSED) {
    const struct pairint *p1 = v1, *p2 = v2;
    if (p1->first != p2->first)
        return p1->first - p2->first;
    return p1->second - p2->second;
}

struct stl_vector3 vector3_sub(struct stl_vector3 a, struct stl_vector3 b) {
    struct stl_vector3 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    return res;
}

float vector3_dot(struct stl_vector3 a, struct stl_vector3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

struct stl_vector3 vector3_cross(struct stl_vector3 a, struct stl_vector3 b) {
    struct stl_vector3 res;
    res.x = a.y*b.z - a.z*b.y;
    res.y = a.z*b.x - a.x*b.z;
    res.z = a.x*b.y - a.y*b.x;
    return res;
}

float vecang(struct stl_vector3 a, struct stl_vector3 b, struct stl_vector3 axis) {
    float lena = sqrtf(vector3_dot(a, a));
    float lenb = sqrtf(vector3_dot(b, b));

    float costheta = vector3_dot(a, b) / (lena*lenb);
    if (costheta < -1)
        costheta = -1;
    else if (costheta > 1)
        costheta = 1;

    float theta = acosf(costheta);
    struct stl_vector3 crsprd = vector3_cross(a, b);
    float projcoeff = vector3_dot(crsprd, axis) / vector3_dot(axis, axis);
    if (projcoeff < 0)
        theta = 2*acos(-1) - theta;

    return theta;
}

void stl_init(struct stl *s) {
    s->nvert = s->ntri = 0;
    s->verts = NULL;
    s->tris = NULL;
}

void stl_read(struct stl *s, FILE *f) {
    char header[81] = "";
    uint32_t ntri;
    struct stl_vector3 vtmp[3];
    uint16_t attr;

    /* Read header (first 80 bytes) */
    fread(header, 1, 80, f);

    /* Read number of triangels (4-bytes unsigned integer) */
    fread(&ntri, 4, 1, f);
    s->ntri = ntri;

    /* Read each triangles and vertices */
    s->tris = calloc(s->ntri, sizeof(struct stl_triangle));
    GTree *vert_tree = g_tree_new_full(cmp_vertex, NULL, g_free, g_free);

    for (int i = 0; i < s->ntri; i++) {
        /* Read normal vector */
        fread(&s->tris[i].n, 4, 3, f);
        /* Read coordinates of 3 vertices */
        for (int j = 0; j < 3; j++)
            fread(vtmp[j].c, 4, 3, f);
        /* Read attribute */
        fread(&attr, 2, 1, f);

        /* Check if vertex is already indexed */
        for (int j = 0; j < 3; j++) {
            int *idx = g_tree_lookup(vert_tree, &vtmp[j]);
            /* If not indexed, insert to `vert_tree` */
            if (!idx) {
                struct stl_vector3 *key = calloc(1, sizeof(struct stl_vector3));
                memcpy(key, &vtmp[j], sizeof(struct stl_vector3));
                int *value = malloc(sizeof(int));
                *value = g_tree_nnodes(vert_tree);
                g_tree_insert(vert_tree, key, value);

                idx = value;
            }
            /* Store the index of vertex in triangle */
            s->tris[i].vertidx[j] = *idx;
        }
    }

    s->nvert = g_tree_nnodes(vert_tree);
    s->verts = calloc(s->nvert, sizeof(struct stl_vector3));

    g_tree_foreach(vert_tree, store_vertex, s->verts);

    g_tree_destroy(vert_tree);

    /* Calculate adjacency of triangles */
    GTree *edge_tree = g_tree_new_full(cmp_pairint, NULL, g_free, g_free);

    for (int i = 0; i < s->ntri; i++) {
        struct pairint etmp;
        for (int j = 0; j < 3; j++) {
            etmp.first = s->tris[i].vertidx[j];
            etmp.second = s->tris[i].vertidx[(j+1)%3];
            if (etmp.first > etmp.second) {
                int tmp = etmp.first;
                etmp.first = etmp.second;
                etmp.second = tmp;
            }

            struct pairint *idx = g_tree_lookup(edge_tree, &etmp);
            if (!idx) {
                struct pairint *key = calloc(1, sizeof(struct pairint));
                memcpy(key, &etmp, sizeof(struct pairint));
                struct pairint *value = calloc(1, sizeof(struct pairint));
                value->first = i;
                value->second = j;
                g_tree_insert(edge_tree, key, value);
            }
            else {
                s->tris[idx->first].adjtriidx[idx->second] = i;
                s->tris[i].adjtriidx[j] = idx->first;
            }
        }
    }

    g_tree_destroy(edge_tree);

    /* Calculate dihedral angles between adjacent triangles */
    for (int i = 0; i < s->ntri; i++) {
        struct stl_vector3 n = s->tris[i].n;
        for (int j = 0; j < 3; j++) {
            struct stl_vector3 nprime = s->tris[s->tris[i].adjtriidx[j]].n;
            struct stl_vector3 e = vector3_sub(
                s->verts[s->tris[i].vertidx[(j+1)%3]],
                s->verts[s->tris[i].vertidx[j]]
            );
            struct stl_vector3 h = vector3_cross(n, e);
            struct stl_vector3 hprime = vector3_cross(e, nprime);
            s->tris[i].dihed[j] = vecang(hprime, h, e);
        }
    }
}

int main(void) {
    FILE *f;
    struct stl stl;

    f = fopen("sphere.stl", "rb");
    if (!f)
        return 0;

    stl_init(&stl);
    stl_read(&stl, f);

    FILE *fout = fopen("stlout.txt", "w");

    fprintf(fout, "# triangles: %zu\n", stl.ntri);
    fprintf(fout, "# vertices: %zu\n", stl.nvert);
    for (int i = 0; i < stl.nvert; i++) {
        fprintf(
            fout,
            "vertex #%4d: %11.6f %11.6f %11.6f\n",
            i, stl.verts[i].x, stl.verts[i].y, stl.verts[i].z
        );
    }
    for (int i = 0; i < stl.ntri; i++) {
        fprintf(
            fout,
            "triangle #%4d: %4d %4d %4d; %4d %4d %4d; %3.0f %3.0f %3.0f\n",
            i, stl.tris[i].vertidx[0], stl.tris[i].vertidx[1], stl.tris[i].vertidx[2],
            stl.tris[i].adjtriidx[0], stl.tris[i].adjtriidx[1], stl.tris[i].adjtriidx[2],
            stl.tris[i].dihed[0]*180/acos(-1), stl.tris[i].dihed[1]*180/acos(-1), stl.tris[i].dihed[2]*180/acos(-1)
        );
    }

    return 0;
}
