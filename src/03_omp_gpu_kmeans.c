// real    0m11.148s
// user    0m43.732s
// sys     0m1.168s

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    FILE *f = fopen("./dataset/train_tratado.csv", "r");
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
    #pragma omp target enter data map(to: data[0:N][0:D], centroides[0:K][0:D]) \
                               map(alloc: grupo[0:N])

    for (int iter = 0; iter < 5000; iter++) {
        // Associação dos pontos
        #pragma omp target teams distribute parallel for \
            map(to: centroides[0:K][0:D]) map(from: grupo[0:N])
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
        for (int i = 0; i < N; i++) {
            int k = grupo[i];
            cont[k]++;
            for (int j = 0; j < D; j++)
                soma[k][j] += data[i][j];
        }

        for (int k = 0; k < K; k++)
            for (int j = 0; j < D; j++)
                centroides[k][j] = soma[k][j] / cont[k];
        #pragma omp target update to(centroides[0:K][0:D])
    }

    // Resultado final
    #pragma omp target exit data map(delete: data, grupo, centroides)
    printf("\nCentroides finais:\n");
    for (int k = 0; k < K; k++) {
        printf("%d: ", k);
        for (int j = 0; j < D; j++)
            printf("%.3f ", centroides[k][j]);
        printf("\n");
    }

    return 0;
}
