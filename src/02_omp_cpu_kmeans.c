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
    int grupo[MAX_POINTS]; // grupo[i] = k significa que o ponto i pertence ao cluster k

    FILE *f = fopen("./dataset/train_tratado.csv", "r");
    if (!f) {
        printf("Erro abrindo arquivo!\n");
        return 1;
    }

    char buffer[4096];

    // Ignorar cabeçalho
    fgets(buffer, sizeof(buffer), f);

    // Ler CSV
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

    // Inicializar centróides
    for (int k = 0; k < K; k++)
        for (int j = 0; j < D; j++)
            centroides[k][j] = data[k][j];

    // K-Means
    for (int iter = 0; iter < 5000; iter++) {

        // 1. Associação dos pontos aos centróides
        #pragma omp parallel for
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

        // 2. Recalcular centróides
        float soma[MAX_K][MAX_D] = {0};
        int cont[MAX_K] = {0};

        // #pragma omp parallel for reduction(+:soma, cont) nao funciona direto com arrays 2D no OMP C

        // usar arrays privados por thread e depois fazer redução serial.

        #pragma omp parallel
        {
            float soma_local[MAX_K][MAX_D] = {0};
            int cont_local[MAX_K] = {0};

            // soma_local = soma dos seus pontos
            // cont_local = quantos pontos do cluster 0, 1, 2,... ela viu

            #pragma omp for
            for (int i = 0; i < N; i++) {
                int k = grupo[i];
                cont_local[k]++; // conta quantos pontos estão nesse cluster k. (será usado para calcular a média)
                for (int j = 0; j < D; j++)
                    soma_local[k][j] += data[i][j]; // soma, para cada dimensão, os valores. (será usado para calcular a média)
            }

            // Redução manual
            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    cont[k] += cont_local[k];
                    for (int j = 0; j < D; j++)
                        soma[k][j] += soma_local[k][j];
                }
            }
        }

        // Atualizar centróides
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
