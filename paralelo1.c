#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
- 1 Thread:
real    0m24.614s
user    0m24.601s
sys     0m0.005s

- 2 Thread:
real    0m14.529s
user    0m29.008s
sys     0m0.011s

- 4 Thread:
real    0m10.439s
user    0m41.632s
sys     0m0.021s

- 8 Thread:
real    0m8.347s
user    1m6.050s
sys     0m0.094s

- 16 Thread:
real    0m10.565s
user    0m46.735s
sys     0m4.462s

- 32 Thread:
real    0m13.114s
user    0m46.073s
sys     0m8.791s
*/

#define MAX_POINTS 20000
#define MAX_D 19
#define MAX_K 5

// Distância Euclidiana
float distancia(float *a, float *b, int d) {
    float soma = 0;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        soma += diff * diff;
    }
    return sqrtf(soma);
}

int main() {
    int N = 0;
    int D = MAX_D;
    int K = MAX_K;

    float data[MAX_POINTS][MAX_D];
    float centroides[MAX_K][MAX_D];
    int grupo[MAX_POINTS];

    FILE *f = fopen("train_tratado.csv", "r");
    if (!f) {
        printf("Erro abrindo arquivo!\n");
        return 1;
    }

    char buffer[4096];

    // Ignorar cabecalho
    fgets(buffer, sizeof(buffer), f);

    // Le CSV separado por vírgulas
    while (fgets(buffer, sizeof(buffer), f) && N < MAX_POINTS) {
        char *ptr = buffer;
        for (int j = 0; j < D; j++) {
            data[N][j] = strtof(ptr, &ptr);
            if (*ptr == ',') ptr++;  // pular vírgula
        }
        N++;
    }

    fclose(f);

    printf("Arquivo lido: %d linhas, %d colunas\n", N, D);

    /*
    // mostrar primeiras linhas
    for (int i = 0; i < 5; i++) {
        printf("Linha %d: ", i);
        for (int j = 0; j < D; j++)
            printf("%f ", data[i][j]);
        printf("\n");
    }
    */

    // Inicializar centróides
    for (int k = 0; k < K; k++)
        for (int j = 0; j < D; j++)
            centroides[k][j] = data[k][j];

    // K-Means
    for (int iter = 0; iter < 5000; iter++) {
        // Associação dos pontos
        #pragma omp parallel for num_threads(32)
        for (int i = 0; i < N; i++) {
            float melhor = 1e30;
            int id = 0;
            for (int k = 0; k < K; k++) {
                float d = distancia(data[i], centroides[k], D);
                if (d < melhor) {
                    melhor = d;
                    id = k;
                }
            }
            grupo[i] = id;
        }

        // Recalcular centróides
        float soma[MAX_K][MAX_D] = {0};
        int cont[MAX_K] = {0};
        #pragma omp parallel for reduction(+:soma, cont) num_threads(32)
        for (int i = 0; i < N; i++) {
            int k = grupo[i];
            cont[k]++;
            for (int j = 0; j < D; j++)
                soma[k][j] += data[i][j];
        }

        for (int k = 0; k < K; k++)
            for (int j = 0; j < D; j++)
                centroides[k][j] = soma[k][j] / cont[k];
    }

    // Resultado final
    printf("\nCentroides finais:\n");
    for (int k = 0; k < K; k++) {
        printf("%d: ", k);
        for (int j = 0; j < D; j++)
            printf("%.3f ", centroides[k][j]);
        printf("\n");
    }

    return 0;
}
