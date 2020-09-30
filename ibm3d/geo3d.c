#include "geo3d.h"

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <glib.h>

const double PI = 3.141592653589793;

struct pairint {
    int first, second;
};

static int cmp_vertex(const void *v1, const void *v2, void *aux G_GNUC_UNUSED);
static int cmp_pairint(const void *v1, const void *v2, void *aux G_GNUC_UNUSED);
static int store_vertex(void *key, void *value, void *aux);
static double vecang(Geo3dVector, Geo3dVector, Geo3dVector);

static void print_vector(const Geo3dVector, const char *);

Geo3dVector Geo3dVector_add(const Geo3dVector a, const Geo3dVector b) {
    Geo3dVector res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    return res;
}

Geo3dVector Geo3dVector_sub(const Geo3dVector a, const Geo3dVector b) {
    Geo3dVector res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    return res;
}

double Geo3dVector_dot(const Geo3dVector a, const Geo3dVector b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Geo3dVector Geo3dVector_crs(const Geo3dVector a, const Geo3dVector b) {
    Geo3dVector res;
    res.x = a.y*b.z - a.z*b.y;
    res.y = a.z*b.x - a.x*b.z;
    res.z = a.x*b.y - a.y*b.x;
    return res;
}

double Geo3dVector_norm(const Geo3dVector v) {
    return sqrt(Geo3dVector_dot(v, v));
}

double Geo3dVector_dist(const Geo3dVector a, const Geo3dVector b) {
    return Geo3dVector_norm(Geo3dVector_sub(a, b));
}

/* Initialize stl object */
void Geo3dPolyhedron_init(Geo3dPolyhedron *poly) {
    poly->num_vertices = poly->num_faces = 0;
    poly->vertex_list = NULL;
    poly->face_list = NULL;
}

/* Read (binary) STL file */
void Geo3dPolyhedron_read_stl(Geo3dPolyhedron *poly, FILE *f) {
    char header[81] = "";
    uint32_t num_faces;
    float coord[3];
    Geo3dVector vtmp[3];
    uint16_t attr;

    /* Read header (first 80 bytes) */
    fread(header, 1, 80, f);

    /* Read number of triangels (next 4-bytes unsigned integer) */
    fread(&num_faces, 4, 1, f);
    poly->num_faces = num_faces;

    /* Read each faces and vertices */
    poly->face_list = calloc(poly->num_faces, sizeof(Geo3dPolyhedronFace));
    GTree *vert_tree = g_tree_new_full(cmp_vertex, NULL, g_free, g_free);

    for (int i = 0; i < poly->num_faces; i++) {
        /* Read normal vector */
        fread(coord, 4, 3, f);
        poly->face_list[i].n.x = coord[0];
        poly->face_list[i].n.y = coord[1];
        poly->face_list[i].n.z = coord[2];
        /* Read coordinates of 3 vertices */
        for (int j = 0; j < 3; j++) {
            fread(coord, 4, 3, f);
            vtmp[j].x = coord[0];
            vtmp[j].y = coord[1];
            vtmp[j].z = coord[2];
        }
        /* Read attribute */
        fread(&attr, 2, 1, f);

        /* Check if vertex is already indexed */
        for (int j = 0; j < 3; j++) {
            int *idx = g_tree_lookup(vert_tree, &vtmp[j]);
            /* If not indexed, insert to `vert_tree` */
            if (!idx) {
                Geo3dVector *key = calloc(1, sizeof(Geo3dVector));
                memcpy(key, &vtmp[j], sizeof(Geo3dVector));
                int *value = malloc(sizeof(int));
                *value = g_tree_nnodes(vert_tree);
                g_tree_insert(vert_tree, key, value);

                idx = value;
            }
            /* Store the index of vertex in triangle */
            poly->face_list[i].vertex_idx[j] = *idx;
        }
    }

    poly->num_vertices = g_tree_nnodes(vert_tree);
    poly->vertex_list = calloc(poly->num_vertices, sizeof(Geo3dVector));

    g_tree_foreach(vert_tree, store_vertex, poly->vertex_list);

    g_tree_destroy(vert_tree);

    /* Calculate adjacency of triangles */
    GTree *edge_tree = g_tree_new_full(cmp_pairint, NULL, g_free, g_free);

    for (int i = 0; i < poly->num_faces; i++) {
        struct pairint etmp;
        for (int j = 0; j < 3; j++) {
            etmp.first = poly->face_list[i].vertex_idx[j];
            etmp.second = poly->face_list[i].vertex_idx[(j+1)%3];
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
                poly->face_list[idx->first].adj_face_idx[idx->second] = i;
                poly->face_list[i].adj_face_idx[j] = idx->first;
            }
        }
    }

    g_tree_destroy(edge_tree);

    /* Calculate dihedral angles between adjacent faces */
    for (int i = 0; i < poly->num_faces; i++) {
        Geo3dVector n = poly->face_list[i].n;
        for (int j = 0; j < 3; j++) {
            Geo3dVector nprime = poly->face_list[poly->face_list[i].adj_face_idx[j]].n;
            Geo3dVector e = Geo3dVector_sub(
                poly->vertex_list[poly->face_list[i].vertex_idx[(j+1)%3]],
                poly->vertex_list[poly->face_list[i].vertex_idx[j]]
            );
            Geo3dVector h = Geo3dVector_crs(n, e);
            Geo3dVector hprime = Geo3dVector_crs(e, nprime);
            poly->face_list[i].adj_face_angle[j] = vecang(hprime, h, e);
        }
    }
}

Geo3dPlane Geo3dPlane_3pts(Geo3dVector a, Geo3dVector b, Geo3dVector c) {
    Geo3dVector ab = Geo3dVector_sub(b, a);
    Geo3dVector ac = Geo3dVector_sub(c, a);
    Geo3dVector n = Geo3dVector_crs(ab, ac);

    Geo3dPlane res;
    res.n = n;
    res.d = -Geo3dVector_dot(n, a);

    return res;
}

Geo3dVector Geo3dPlane_proj(Geo3dPlane p, Geo3dVector v) {
    double k = (p.a*v.x+p.b*v.y+p.c*v.z+p.d) / (p.a*p.a+p.b*p.b+p.c*p.c);
    Geo3dVector res;
    res.x = v.x - k*p.a;
    res.y = v.y - k*p.b;
    res.z = v.z - k*p.c;
    return res;
}

double Geo3dPolyhedron_sgndist(Geo3dPolyhedron *poly, Geo3dVector v) {
    double res = INFINITY;

    for (int i = 0; i < poly->num_faces; i++) {
        Geo3dPolyhedronFace *face = &poly->face_list[i];

        Geo3dVector a = poly->vertex_list[face->vertex_idx[0]];
        Geo3dVector b = poly->vertex_list[face->vertex_idx[1]];
        Geo3dVector c = poly->vertex_list[face->vertex_idx[2]];

        Geo3dVector ab = Geo3dVector_sub(b, a);
        Geo3dVector bc = Geo3dVector_sub(c, b);
        Geo3dVector ca = Geo3dVector_sub(a, c);

        Geo3dPlane p = Geo3dPlane_3pts(a, b, c);
        Geo3dVector h = Geo3dPlane_proj(p, v);

        // print_vector(face->n, "face normal");
        // print_vector(h, "h");

        double area = 0.5 * Geo3dVector_dot(Geo3dVector_crs(ab, bc), face->n);
        double ka = 0.5 * Geo3dVector_dot(Geo3dVector_crs(Geo3dVector_sub(b, h), bc), face->n) / area;
        double kb = 0.5 * Geo3dVector_dot(Geo3dVector_crs(Geo3dVector_sub(c, h), ca), face->n) / area;
        double kc = 1 - ka - kb;

        // printf("area: %lf, ka: %lf, kb: %lf, kc: %lf\n", area, ka, kb, kc);

        /* h is on the face */
        if (ka > 0 && kb > 0 && kc > 0) {
            double dist = Geo3dVector_dist(v, h);
            // printf("dist: %lf\n", dist);
            if (Geo3dVector_dot(Geo3dVector_sub(v, h), face->n) < 0) {
                dist = -dist;
            }
            if (fabs(dist) < fabs(res)) {
                res = dist;
            }
            continue;
        }

        double tab = Geo3dVector_dot(Geo3dVector_sub(v, a), ab) / Geo3dVector_dot(ab, ab);
        double tbc = Geo3dVector_dot(Geo3dVector_sub(v, b), bc) / Geo3dVector_dot(bc, bc);
        double tca = Geo3dVector_dot(Geo3dVector_sub(v, c), ca) / Geo3dVector_dot(ca, ca);

        Geo3dVector habv, hbcv, hcav;

        habv.x = v.x - (1-tab)*a.x - tab*b.x;
        habv.y = v.y - (1-tab)*a.y - tab*b.y;
        habv.z = v.z - (1-tab)*a.z - tab*b.z;

        hbcv.x = v.x - (1-tbc)*b.x - tbc*c.x;
        hbcv.y = v.y - (1-tbc)*b.y - tbc*c.y;
        hbcv.z = v.z - (1-tbc)*b.z - tbc*c.z;

        hcav.x = v.x - (1-tca)*c.x - tca*a.x;
        hcav.y = v.y - (1-tca)*c.y - tca*a.y;
        hcav.z = v.z - (1-tca)*c.z - tca*a.z;

        double thetaab = 1.5*PI - vecang(face->n, habv, ab);
        double thetabc = 1.5*PI - vecang(face->n, hbcv, bc);
        double thetaca = 1.5*PI - vecang(face->n, hcav, ca);

        bool neara = false, nearb = false, nearc = false;

        if (ka < 0) {
            if (tbc < 0) {
                nearb = true;
            }
            else if (tbc > 1) {
                nearc = true;
            }
            else {
                double dist = Geo3dVector_norm(hbcv);
                if (face->adj_face_angle[1] > thetabc) {
                    dist = -dist;
                }
                if (fabs(dist) < fabs(res)) {
                    res = dist;
                }
                continue;
            }
        }
        else if (kb < 0) {
            if (tca < 0) {
                nearc = true;
            }
            else if (tca > 1) {
                neara = true;
            }
            else {
                double dist = Geo3dVector_norm(hcav);
                if (face->adj_face_angle[2] > thetaca) {
                    dist = -dist;
                }
                if (fabs(dist) < fabs(res)) {
                    res = dist;
                }
                continue;
            }
        }
        else if (kc < 0) {
            if (tab < 0) {
                neara = true;
            }
            else if (tab > 1) {
                nearb = true;
            }
            else {
                double dist = Geo3dVector_norm(habv);
                if (face->adj_face_angle[0] > thetaab) {
                    dist = -dist;
                }
                if (fabs(dist) < fabs(res)) {
                    res = dist;
                }
                continue;
            }
        }

        if (neara && !nearb && !nearc) {
            double dist = Geo3dVector_dist(v, a);
            if (face->adj_face_angle[2] > thetaca && face->adj_face_angle[0] > thetaab) {
                dist = -dist;
            }
            if (fabs(dist) < fabs(res)) {
                res = dist;
            }
        }
        else if (!neara && nearb && !nearc) {
            double dist = Geo3dVector_dist(v, b);
            if (face->adj_face_angle[0] > thetaab && face->adj_face_angle[1] > thetabc) {
                dist = -dist;
            }
            if (fabs(dist) < fabs(res)) {
                res = dist;
            }
        }
        else if (!neara && !nearb && nearc) {
            double dist = Geo3dVector_dist(v, c);
            if (face->adj_face_angle[1] > thetabc && face->adj_face_angle[2] > thetaca) {
                dist = -dist;
            }
            if (fabs(dist) < fabs(res)) {
                res = dist;
            }
        }
        else {
            printf("FATAL ERROR: %d %d %d\n", (int)neara, (int)nearb, (int)nearc);
        }
    }

    return res;
}

static int cmp_vertex(const void *v1, const void *v2, void *aux G_GNUC_UNUSED) {
    const Geo3dVector *vt1 = v1, *vt2 = v2;
    if (vt1->x != vt2->x)
        return vt1->x < vt2->x ? -1 : 1;
    if (vt1->y != vt2->y)
        return vt1->y < vt2->y ? -1 : 1;
    if (vt1->z != vt2->z)
        return vt1->z < vt2->z ? -1 : 1;
    return 0;
}

static int store_vertex(void *key, void *value, void *aux) {
    Geo3dVector *vertex_list = aux;
    int idx = *(int *)value;
    memcpy(&vertex_list[idx], key, sizeof(Geo3dVector));
    return 0;
}

static int cmp_pairint(const void *v1, const void *v2, void *aux G_GNUC_UNUSED) {
    const struct pairint *p1 = v1, *p2 = v2;
    if (p1->first != p2->first)
        return p1->first - p2->first;
    return p1->second - p2->second;
}

static double vecang(Geo3dVector a, Geo3dVector b, Geo3dVector axis) {
    double lena = Geo3dVector_norm(a);
    double lenb = Geo3dVector_norm(b);

    double costheta = Geo3dVector_dot(a, b) / (lena*lenb);
    if (costheta < -1)
        costheta = -1;
    else if (costheta > 1)
        costheta = 1;

    double theta = acos(costheta);
    Geo3dVector crsprd = Geo3dVector_crs(a, b);
    double projcoeff = Geo3dVector_dot(crsprd, axis) / Geo3dVector_dot(axis, axis);
    if (projcoeff < 0)
        theta = 2*acos(-1) - theta;

    return theta;
}

static void print_vector(const Geo3dVector v, const char *name) {
    printf("%s: (%lf, %lf, %lf)\n", name, v.x, v.y, v.z);
}
