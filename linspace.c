#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("not enough #args\n");
        return 0;
    }

    int num_segs = argc / 2 - 1;
    int num_pts = 0;

    printf("%.8lf ", atof(argv[1]));
    for (int i = 0; i < num_segs; i++) {
        double a = atof(argv[2*i+1]);
        double b = atof(argv[2*i+3]);
        int n = atoi(argv[2*i+2]);
        num_pts += n;
        for (int j = 1; j <= n; j++) {
            printf("%.8lf ", a + (b - a) * j / n);
        }
    }
    printf("\n%d\n", num_pts);

    return 0;
}