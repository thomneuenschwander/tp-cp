// real    0m0.768s
// user    0m0.416s
// sys     0m0.314s

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_POINTS 20000
#define MAX_D 19
#define MAX_K 5

// calcula o grupo do ponto i. Cada da thread (CUDA core) executa esse kernel para um ponto i diferente
__global__ void associaPontos(float *data, float *centroides, int *grupo, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //  Cada thread calcula um índice único i que corresponde a um ponto de dado específico (data[i]). Se o índice exceder N, a thread termina. 
    if (i >= N) return; // calcular um índice global único que varia de 0 a $N-1$

    // threadIdx.x -> Índice da thread dentro do bloco.
    // blockDim.x -> Número total de threads por bloco.
    // blockIdx.x -> Índice do bloco dentro da grade (grid).

    float melhor = 1e30;
    int melhorK = 0;

    for (int k = 0; k < K; k++) { // encontrar o centróide mais próximo de data[i]
        float soma = 0;
        for (int j = 0; j < D; j++) {
            float diff = data[i * D + j] - centroides[k * D + j];
            soma += diff * diff;
        }
        float dist = sqrtf(soma);

        if (dist < melhor) {
            melhor = dist;
            melhorK = k;
        }
    }

    grupo[i] = melhorK;
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
    fgets(buffer, sizeof(buffer), f); // pular cabeçalho

    while (fgets(buffer, sizeof(buffer), f) && N < MAX_POINTS) {
        char *ptr = buffer;
        for (int j = 0; j < D; j++) {
            data[N][j] = strtof(ptr, &ptr);
            if (*ptr == ',') ptr++;
        }
        N++;
    }
    fclose(f);

    printf("Arquivo lido: %d linhas, %d colunas\n", N, D);

    // Inicializa centróides usando os primeiros K pontos
    for (int k = 0; k < K; k++)
        for (int j = 0; j < D; j++)
            centroides[k][j] = data[k][j];

   
    float *d_data, *d_centroides;
    int *d_grupo;

    cudaMalloc(&d_data, N * D * sizeof(float)); // alocar na vRAM (GPU)
    cudaMalloc(&d_centroides, K * D * sizeof(float));
    cudaMalloc(&d_grupo, N * sizeof(int));

    cudaMemcpy(d_data, data, N * D * sizeof(float), cudaMemcpyHostToDevice); // cp dados para vRAM

    int threads = 256;
    int blocos = (N + threads - 1) / threads;

    for (int iter = 0; iter < 500; iter++) {

        cudaMemcpy(d_centroides, centroides, K * D * sizeof(float), cudaMemcpyHostToDevice); // Os centróides calculados na CPU (no loop anterior) são copiados para a GPU antes de cada nova associação.

        associaPontos<<<blocos, threads>>>(d_data, d_centroides, d_grupo, N, D, K); // lança o kernel (associaPontos) na GPU,  dividindo o trabalho de N pontos entre blocos e threads.
        cudaDeviceSynchronize(); // Garante que a CPU espere que todos os threads da GPU terminem a execução do kernel antes de prosseguir.

        cudaMemcpy(grupo, d_grupo, N * sizeof(int), cudaMemcpyDeviceToHost);

        // Recálculo de Centróides na CPU -> (O Gargalo Serial)
        float soma[MAX_K][MAX_D] = {0};
        int cont[MAX_K] = {0};

        for (int i = 0; i < N; i++) {
            int g = grupo[i];
            cont[g]++;
            for (int j = 0; j < D; j++)
                soma[g][j] += data[i][j];
        }

        for (int k = 0; k < K; k++)
            for (int j = 0; j < D; j++)
                centroides[k][j] = soma[k][j] / cont[k];
    }

    cudaFree(d_data);
    cudaFree(d_centroides);
    cudaFree(d_grupo);

    // Resultado
    printf("\nCentroides finais:\n");
    for (int k = 0; k < K; k++) {
        printf("%d: ", k);
        for (int j = 0; j < D; j++)
            printf("%.3f ", centroides[k][j]);
        printf("\n");
    }

    return 0;
}
