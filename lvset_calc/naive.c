#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double dist(double a[static 2], double b[static 2]) {
    return sqrt((b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]));
}

double vecang(double a[static 2], double b[static 2]) {
    double lena = sqrt(a[0]*a[0]+a[1]*a[1]);
    double lenb = sqrt(b[0]*b[0]+b[1]*b[1]);

    double theta = acos((a[0]*b[0]+a[1]*b[1]) / (lena*lenb));
    double crsprod = a[0]*b[1] - a[1]*b[0];
    if (crsprod < 0) {
        theta = 2*acos(-1) - theta;
    }

    return theta;
}

int main(int argc, char **argv) {
    FILE *fp;

    int n;
    double (*v)[2];

    fp = fopen("polygon.in", "r");
    if (fp) {
        fscanf(fp, "%d", &n);
        v = calloc(n, sizeof(double [2]));
        for (int i = 0; i < n; i++) {
            fscanf(fp, "%lf %lf", &v[i][0], &v[i][1]);
        }
        fclose(fp);
    }

    int Nx, Ny;
    double *xf, *yf, *xc, *yc;

    fp = fopen("mesh.in", "r");
    if (fp) {
        fscanf(fp, "%*s %d", &Nx);
        xf = calloc(Nx+1, sizeof(double));
        for (int i = 0; i <= Nx; i++) {
            fscanf(fp, "%lf", &xf[i]);
        }

        fscanf(fp, "%*s %d", &Ny);
        yf = calloc(Ny+1, sizeof(double));
        for (int i = 0; i <= Ny; i++) {
            fscanf(fp, "%lf", &yf[i]);
        }

        fclose(fp);
    }

    xc = calloc(Nx+2, sizeof(double));
    yc = calloc(Ny+2, sizeof(double));
    for (int i = 1; i <= Nx; i++) {
        xc[i] = (xf[i-1] + xf[i]) / 2;
    }
    for (int i = 1; i <= Ny; i++) {
        yc[i] = (yf[i-1] + yf[i]) / 2;
    }

    double *theta = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double a[2] = {v[(i+1)%n][0]-v[i][0], v[(i+1)%n][1]-v[i][1]};
        double b[2] = {v[(i+n-1)%n][0]-v[i][0], v[(i+n-1)%n][1]-v[i][1]};
        theta[i] = vecang(a, b);
    }

    double (*lvset)[Ny+2] = calloc(Nx+2, sizeof(double [Ny+2]));
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            lvset[i][j] = INFINITY;
        }
    }

    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            for (int k = 0; k < n; k++) {
                double p[2] = {xc[i], yc[j]};
                double s[2] = {v[(k+1)%n][0]-v[k][0], v[(k+1)%n][1]-v[k][1]};
                double pb[2] = {v[(k+1)%n][0]-xc[i], v[(k+1)%n][1]-yc[j]};
                double t = (s[0]*pb[0]+s[1]*pb[1]) / (s[0]*s[0]+s[1]*s[1]);
                double d;
                if (t > 1) {
                    d = dist(v[k], p);
                    double ap[2] = {p[0]-v[k][0], p[1]-v[k][1]};
                    double alpha = vecang(s, ap);
                    if (alpha < theta[k]) {
                        d = -d;
                    }
                }
                else if (t < 0) {
                    d = dist(v[(k+1)%n], p);
                    double alpha = vecang((double [2]){-pb[0], -pb[1]}, (double [2]){-s[0], -s[1]});
                    if (alpha < theta[(k+1)%n]) {
                        d = -d;
                    }
                }
                else {
                    double h[2] = {t*v[k][0]+(1-t)*v[(k+1)%n][0], t*v[k][1]+(1-t)*v[(k+1)%n][1]};
                    d = dist(h, p);
                    double ap[2] = {p[0]-v[k][0], p[1]-v[k][1]};
                    if (vecang(s, ap) < acos(-1)) {
                        d = -d;
                    }
                }

                if (fabs(d) < fabs(lvset[i][j])) {
                    lvset[i][j] = d;
                }
            }
        }
    }

    fp = fopen("lvset.out", "w");
    if (fp) {
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                fprintf(fp, "%.6lf ", lvset[i][j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    return 0;
}