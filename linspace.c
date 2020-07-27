#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("please specify #points\n");
        return 0;
    }
    if (argc == 3) {
        printf("please specify both left and right end\n");
        return 0;
    }

    int n = atoi(argv[1]);
    double a, b;
    if (argc >= 4) {
        a = atof(argv[2]);
        b = atof(argv[3]);
    }
    else {
        a = 0;
        b = 1;
    }

    for (int i = 0; i < n; i++) {
        printf("%.8lf ", a + (b - a) * i / (n-1));
    }
    printf("\n");

    return 0;
}