import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict

def plot_openmp_CPU_parallel_performance(threads: list, tempos_reais: list):
    threads_np = np.array(threads)
    tempos_np = np.array(tempos_reais)
    
    tempo_serial = tempos_np[0]
    aceleracao = tempo_serial / tempos_np
    
    melhor_tempo_index = np.argmin(tempos_np)
    melhor_thread = threads_np[melhor_tempo_index]
    melhor_tempo = tempos_np[melhor_tempo_index]
    
    diretorio_saida = "./plots"
    os.makedirs(diretorio_saida, exist_ok=True)

    plt.figure(figsize=(8, 6)) # Tamanho já está bom

    plt.plot(threads_np, tempos_np, marker='o', linestyle='-', color='blue', label='Tempo Real (s)')
    
    plt.plot(melhor_thread, melhor_tempo, marker='*', markersize=15, color='red', 
             label=f'Pico de Desempenho ({melhor_thread}T)')
    
    plt.title('Escalabilidade do K-Means - OpenMP CPU', fontsize=16)
    plt.xlabel('Número de Threads', fontsize=12)
    plt.ylabel('Tempo de Execução Real (segundos)', fontsize=12)
    
    plt.xticks(threads_np)
    
    plt.text(melhor_thread, melhor_tempo, 
             f' {melhor_tempo:.3f}s', 
             verticalalignment='bottom', 
             horizontalalignment='left', 
             color='red', 
             fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    nome_arquivo = "kmeans_escalabilidade_tempo_02_cpu.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)
    
    plt.savefig(caminho_completo) 
    plt.close()

    print("\n--- Métricas de Desempenho ---")
    print(f"Gráfico salvo em: {caminho_completo}")
    print(f"Tempo Serial (1T): {tempo_serial:.3f} s")
    print(f"Melhor Desempenho em: {melhor_thread} Threads ({melhor_tempo:.3f} s)")
    print(f"Aceleração Máxima (Speedup): {aceleracao[melhor_tempo_index]:.2f}x")
    

threads_testadas = [1, 2, 4, 6, 8, 16, 32]
tempos_reais_segundos = [
    27.301,  # 1T
    13.751,  # 2T
    7.013,  # 4T
    4.951,  # 6T -> Melhor Tempo
    10.050,  # 8T
    6.987,  # 16T
    6.624   # 32T
]

plot_openmp_CPU_parallel_performance(threads_testadas, tempos_reais_segundos)


def plot_comparison_performance(tempos_abordagens: Dict[str, float]):
    """
    Gera um gráfico de barras comparando o tempo de execução 
    de diferentes abordagens de paralelismo (Serial, OpenMP CPU, OpenMP GPU, CUDA).

    Args:
        tempos_abordagens (Dict[str, float]): Dicionário com o nome da abordagem e seu tempo real em segundos.
    """
    
    abordagens = list(tempos_abordagens.keys())
    tempos_np = np.array(list(tempos_abordagens.values()))
    
    # Define as cores
    cores = ['gray', 'green', 'orange', 'red']
    
    # Encontra o melhor tempo (excluindo CUDA, se desejado, mas vamos incluir tudo)
    melhor_tempo_index = np.argmin(tempos_np)
    melhor_tempo = tempos_np[melhor_tempo_index]
    
    diretorio_saida = "./plots"
    os.makedirs(diretorio_saida, exist_ok=True)
    
    plt.figure(figsize=(10, 6))

    # Cria o gráfico de barras
    barras = plt.bar(abordagens, tempos_np, color=cores, alpha=0.8)
    
    # Configuração visual e rótulos
    plt.title('Comparação de Desempenho do K-Means (Tempo Real)', fontsize=16)
    plt.xlabel('Abordagem de Paralelismo', fontsize=12)
    plt.ylabel('Tempo de Execução Real (segundos)', fontsize=12)
    
    # Adiciona os valores acima de cada barra
    for i, barra in enumerate(barras):
        yval = barra.get_height()
        
        # Coloca a estrela vermelha no pico de desempenho
        if i == melhor_tempo_index:
            plt.text(barra.get_x() + barra.get_width()/2, yval + 0.5, '⭐', 
                     ha='center', va='bottom', fontsize=18, color='red')
            
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.1, 
                 f'{yval:.3f}s', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, max(tempos_np) * 1.2) # Ajusta o limite Y para caber os rótulos
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    nome_arquivo = "kmeans_comparacao_desempenho_03.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)
    
    plt.savefig(caminho_completo) 
    plt.close()

    print(f"\nGráfico de comparação salvo em: {caminho_completo}")
    print(f"O melhor tempo foi: {abordagens[melhor_tempo_index]} com {melhor_tempo:.3f} s.")
    
    
tempos_de_teste = {
    "Serial": 27.650,
    "OpenMP CPU (6T)": 4.951,
    "OpenMP GPU": 11.148,
    "CUDA (GPU)": 0.768
}
    
plot_comparison_performance(tempos_de_teste)