#include<stdio.h>
#include<stdlib.h>

int** build_matrix(char *path, int* x, int* y) {
    int i;
    int j;

    FILE *file;
    file = fopen(path, "r");
    int lenX;
    int lenY;
    char p;
    fscanf(file, "%d", &lenX);
    fscanf(file, "%c", &p);
    fscanf(file, "%d", &lenY);

    if (lenY == 0) {
        lenY = 1;
    }

    int** mat;
    mat = malloc(lenX * sizeof(int *));
    for(i = 0; i < lenX; i++) {
        mat[i] = malloc(lenY * sizeof(int));
    }
    
    printf("%d", lenX);
    printf("%d", lenY);
    for(i = 0; i < lenX; i++) {
        for(j = 0; j < lenY; j++) {
            fscanf(file, "%d", &mat[i][j]);
            fscanf(file, "%c", &p);
            printf("%d ", mat[i][j]);
            if (p == '\n') {
                break;
            }
        }
    }

    *x = lenX;
    if (lenY == 1) {
        *y = lenY - 1;
    } else {
        *y = lenY;
    }

    fclose(file);
    return mat;
}

int main(void) {
    int a;
    int b;
    build_matrix("../data/val_target.txt", &a, &b);
    printf("%d, %d", a, b);
    return 0;
}
