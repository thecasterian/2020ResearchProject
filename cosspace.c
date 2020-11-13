#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("please specify #points\n");
        return 0;
    }

    double a = atof(argv[1]);
    int n = atoi(argv[2]);
    double b = atof(argv[3]);
    for (int i = 0; i <= n; i++) {
        double x = acos(-1) * i / n;
        printf("%.8lf ", (b-a)*(1-cos(x))/2+a);
    }
    printf("\n");

    return 0;
}