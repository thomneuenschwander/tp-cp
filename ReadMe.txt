Trabalho de Computação Paralela

Grupo: 
- Henrique Lara
- Lívia Xavier
- Thomas Neuenschwander


A base de dados utilizada vem do Kaggle e ela tem dados sobre quanto tempo um pet leva para ser adotado.
 O atributo “AdoptionSpeed” é a classe original do problema, dividida em categorias que indicam em quantos dias 
 o animal foi adotado:

0 – No mesmo dia em que foi listado

1 – Entre 1 e 7 dias
2 – Entre 8 e 30 dias
3 – Entre 31 e 90 dias
4 – Não foi adotado após 100 dias

Como o foco não é classificação, mas sim observar a distribuição natural dos dados usando K-Means, removemos
 o atributo AdoptionSpeed para evitar que o algoritmo fosse influenciado por uma variável que representa o rótulo 
 da tarefa original. Além disso, também excluímos atributos nominais ou que não contribuíam para o agrupamento,
  como o nome do pet. 

0) Sequencial - (Arquivo "sequencial.c"):

real    0m25.398s
user    0m25.372s
sys     0m0.014s

(Foi compilado com: 
gcc sequencial.c -o sequencial -lm 
time ./sequencial  )

i) OpenMP para multicore (CPU) - (Arquivo "paralelo1.c"):

Usamos reduction porque, dentro do loop, fazemos a soma das coordenadas para calcular os centróides. 
Se várias threads tentassem atualizar essas somas ao mesmo tempo, os valores seriam sobrescritos e o resultado
 sairia errado. Com o reduction, cada thread trabalha com uma cópia própria das variáveis soma e cont,
  faz suas operações de forma independente e, no final, todas essas cópias são agrupadas/combinadas.

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

(Foi compilado com: 
gcc -fopenmp paralelo1.c -o paralelo1 -lm 
time ./paralelo1   )

ii) OpenMP para GPU - (Arquivo "paralelo2.c"):

Primeiramente, para a paralelização acontecer são enviados para a GPU apenas os dados que serão utilizados durante as iterações.
Após isso, o comando "target teams distribute parallel for" faz com que a GPU crie vários grupos (teams) contendo múltiplas threads,
 que ficarão responsáveis por calcular a distância entre os pontos e os centróides. 

Os centróides são recalculados na CPU, então eles precisam ser enviados novamente para a GPU a cada iteração através do comando "target update to".
O vetor grupo[], gerado na GPU, é copiado de volta para a CPU dentro do próprio loop com "map(from: grupo[0:N])" e por isso
o comando "target exit data" apenas libera a memória da GPU com "map(delete: ...)", sem copiar nada de volta.

real    0m3.376s
user    0m11.339s
sys     0m1.472s

(Foi compilado com: 
gcc paralelo2.c -O3 -fopenmp -o paralelo2 -lm
time ./paralelo2  )

iii) CUDA para GPU - (Arquivo "paralelo3.c"):


