#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("please specify #points\n");
        return 0;
    }

    int n = atoi(argv[1]);
    for (int i = 0; i < n; i++) {
        printf("%.8lf ", (double)i / (n-1));
    }
    printf("\n");

    return 0;
}