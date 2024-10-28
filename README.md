# Algoritmo Genético para o Passeio do Cavalo no Tabuleiro de Xadrez

## Sumário

1. [Introdução](#introdução)
2. [Algoritmos Genéticos (AG)](#algoritmos-genéticos-ag)
   - [Visão Geral](#visão-geral)
   - [Componentes de um AG](#componentes-de-um-ag)
     - [Representação dos Indivíduos (Cromossomos)](#representação-dos-indivíduos-cromossomos)
     - [População Inicial](#população-inicial)
     - [Função de Fitness](#função-de-fitness)
     - [Seleção](#seleção)
     - [Crossover (Recombinação)](#crossover-recombinação)
     - [Mutação](#mutação)
3. [Problema do Passeio do Cavalo](#problema-do-passeio-do-cavalo)
   - [Descrição do Problema](#descrição-do-problema)
   - [Abordagem com Algoritmo Genético](#abordagem-com-algoritmo-genético)
4. [Implementação do AG para o Passeio do Cavalo](#implementação-do-ag-para-o-passeio-do-cavalo)
   - [Estrutura do Código](#estrutura-do-código)
   - [Funções Principais](#funções-principais)
     - [Obter Movimentos Possíveis](#obter-movimentos-possíveis)
     - [Função de Fitness](#função-de-fitness-1)
     - [Inicializar Indivíduo](#inicializar-indivíduo)
     - [Inicializar População](#inicializar-população)
     - [Selecionar Elite](#selecionar-elite)
     - [Crossover](#crossover)
     - [Mutação](#mutação-1)
     - [Algoritmo Genético (AG)](#algoritmo-genético-ag)
     - [Visualização do Tabuleiro](#visualização-do-tabuleiro)
     - [Exibir Movimentos](#exibir-movimentos)
   - [Execução do Algoritmo](#execução-do-algoritmo)
5. [Análise e Resultados](#análise-e-resultados)
   - [Convergência](#convergência)
   - [Eficiência](#eficiência)
   - [Revisitações](#revisitações)
6. [Conclusão](#conclusão)
7. [Referências](#referências)

---

## Introdução

O **Passeio do Cavalo** é um clássico problema de xadrez que desafia a capacidade de encontrar uma sequência de movimentos de um cavalo de forma que ele visite cada uma das 64 casas do tabuleiro exatamente uma vez. Resolver este problema pode ser feito através de diversas abordagens algorítmicas, incluindo algoritmos genéticos (AG).

Os **Algoritmos Genéticos** são métodos de otimização inspirados na seleção natural e na genética. Eles são eficazes para resolver problemas complexos e de grande escala, onde métodos exatos podem ser ineficientes ou impraticáveis.

Este documento detalha a implementação de um Algoritmo Genético para resolver o problema do Passeio do Cavalo no tabuleiro de xadrez, explicando os conceitos teóricos e a aplicação prática através de um código Python.

---

## Algoritmos Genéticos (AG)

### Visão Geral

Os **Algoritmos Genéticos (AG)** são uma classe de algoritmos de busca e otimização inspirados nos processos de evolução natural. Eles operam em uma população de indivíduos, onde cada indivíduo representa uma solução potencial para o problema em questão.

O processo evolutivo nos AGs envolve seleção, reprodução e mutação para gerar novas gerações de indivíduos, com o objetivo de melhorar a qualidade das soluções ao longo do tempo.

### Componentes de um AG

Para entender como os AGs funcionam, é fundamental conhecer seus componentes principais:

#### Representação dos Indivíduos (Cromossomos)

- **Cromossomo:** Representa uma solução potencial para o problema. Pode ser uma sequência de bits, números, ou qualquer estrutura que codifique a solução.
  
- **Genes:** Cada elemento do cromossomo, representando uma parte específica da solução.

**No contexto do Passeio do Cavalo:**
  
- Cada indivíduo é uma lista de posições no tabuleiro, representando a sequência de movimentos do cavalo.
  
- Cada gene é uma posição única no tabuleiro (0 a 63).

#### População Inicial

- **População:** Conjunto de indivíduos (soluções) que serão evoluídos ao longo das gerações.

- **Inicialização:** A população inicial é geralmente gerada aleatoriamente para garantir diversidade genética.

**No Passeio do Cavalo:**
  
- A população inicial consiste em várias sequências aleatórias de movimentos válidos do cavalo, começando de posições aleatórias no tabuleiro.

#### Função de Fitness

- **Fitness:** Medida da qualidade de um indivíduo. Quanto maior o fitness, melhor a solução.

- **Objetivo:** Maximizar (ou minimizar) a função de fitness para encontrar a melhor solução possível.

**No Passeio do Cavalo:**
  
- A função de fitness conta o número de posições únicas visitadas pelo cavalo na sequência de movimentos.

#### Seleção

- **Seleção:** Processo de escolher quais indivíduos da população atual serão usados para gerar a próxima geração.

- **Métodos Comuns:** Roleta, torneio, seleção elitista.

**No Passeio do Cavalo:**
  
- A seleção elitista é utilizada, onde os melhores indivíduos (com maior fitness) são selecionados para serem pais na próxima geração.

#### Crossover (Recombinação)

- **Crossover:** Combinação de genes de dois pais para produzir novos filhos.

- **Tipos:** Single-point, multi-point, uniform.

**No Passeio do Cavalo:**
  
- O crossover combina sequências de movimentos de dois pais, garantindo que os filhos resultantes não revisitem posições já visitadas.

#### Mutação

- **Mutação:** Alteração aleatória de genes em um indivíduo para introduzir diversidade.

- **Objetivo:** Evitar a convergência prematura e explorar novas regiões do espaço de soluções.

**No Passeio do Cavalo:**
  
- A mutação altera um ponto na sequência de movimentos, substituindo-o por um movimento válido que não tenha sido visitado anteriormente.

---

## Problema do Passeio do Cavalo

### Descrição do Problema

O **Passeio do Cavalo** (Knight's Tour) é um problema no qual se busca uma sequência de movimentos de um cavalo de xadrez de forma que ele visite cada uma das 64 casas do tabuleiro exatamente uma vez.

Existem duas variantes:

1. **Passeio Fechado:** O cavalo retorna à casa inicial no final do passeio.
2. **Passeio Aberto:** O cavalo não necessariamente retorna à casa inicial.

Este trabalho foca no **Passeio Aberto**.

### Abordagem com Algoritmo Genético

Aplicar um **Algoritmo Genético** ao Passeio do Cavalo envolve:

1. **Representação dos Indivíduos:** Sequência de movimentos do cavalo.
2. **População Inicial:** Diversas sequências aleatórias de movimentos válidos.
3. **Fitness:** Número de casas únicas visitadas.
4. **Seleção:** Escolha dos melhores indivíduos para reprodução.
5. **Crossover e Mutação:** Geração de novos indivíduos através de recombinação e alterações aleatórias.
6. **Iteração:** Repetir o processo até encontrar uma sequência que visite todas as casas.

---

## Implementação do AG para o Passeio do Cavalo

A seguir, detalhamos a implementação do Algoritmo Genético para resolver o Passeio do Cavalo, explicando cada parte do código.

### Estrutura do Código

O código está organizado em diversas funções, cada uma responsável por uma parte específica do AG:

1. **Configurações e Constantes**
2. **Funções de Utilidade**
3. **Funções Genéticas (Fitness, Seleção, Crossover, Mutação)**
4. **Funções de Inicialização**
5. **Função Principal (main)**
6. **Visualização e Registro de Revisitações**

### Funções Principais

#### Obter Movimentos Possíveis

```python
def obter_movimentos_possiveis(pos_atual: int, tamanho: int = TAMANHO_TABULEIRO) -> List[int]:
    """
    Calcula as posições para as quais um cavalo pode se mover em um tabuleiro de xadrez
    representado unidimensionalmente.
    """
    lado: int = int(np.sqrt(tamanho))
    if lado * lado != tamanho:
        raise ValueError(
            "O tamanho do tabuleiro deve ser um quadrado perfeito (e.g., 64 para 8x8)."
        )

    linha: int = pos_atual // lado
    coluna: int = pos_atual % lado

    novas_linhas: np.ndarray = linha + POSSIVEIS_MOVIMENTOS_CAVALO[:, 0]
    novas_colunas: np.ndarray = coluna + POSSIVEIS_MOVIMENTOS_CAVALO[:, 1]

    validos: np.ndarray = (
        (novas_linhas >= 0)
        & (novas_linhas < lado)
        & (novas_colunas >= 0)
        & (novas_colunas < lado)
    )

    novos_indices: np.ndarray = novas_linhas[validos] * lado + novas_colunas[validos]

    return novos_indices.tolist()
```

**Descrição:**

- **Objetivo:** Determinar todas as posições para as quais o cavalo pode se mover a partir de uma posição atual no tabuleiro.
  
- **Parâmetros:**
  - `pos_atual`: Posição atual do cavalo no tabuleiro (0 a 63).
  - `tamanho`: Tamanho total do tabuleiro (default 64 para 8x8).
  
- **Processo:**
  1. Calcula as coordenadas (linha, coluna) a partir do índice unidimensional.
  2. Aplica todos os movimentos possíveis do cavalo para obter novas coordenadas.
  3. Filtra os movimentos que resultam em posições válidas dentro do tabuleiro.
  4. Converte as coordenadas válidas de volta para índices unidimensionais.

#### Função de Fitness

```python
def fitness(individual: List[int]) -> int:
    """
    Avalia a aptidão de um indivíduo.

    A aptidão é definida como o número de posições únicas visitadas pelo cavalo.
    """
    return len(set(individual))
```

**Descrição:**

- **Objetivo:** Medir a qualidade de um indivíduo na população.
  
- **Processo:** Conta o número de posições únicas visitadas pelo cavalo na sequência de movimentos.

- **Interpretação:** Quanto maior o número de posições únicas, maior a aptidão do indivíduo.

#### Verificar Revisitas

```python
def verificar_revisitas(individual: List[int]) -> bool:
    """
    Verifica se o indivíduo possui revisitações.

    Retorna True se houver revisitações, False caso contrário.
    """
    visitados = set()
    for pos in individual:
        if pos in visitados:
            revisitas_registradas.append(pos)
            return True
        visitados.add(pos)
    return False
```

**Descrição:**

- **Objetivo:** Detectar se há revisitações (visita repetida a uma posição) no passeio do cavalo.

- **Processo:**
  1. Percorre a sequência de posições visitadas.
  2. Se uma posição já foi visitada, registra a revisitação e retorna `True`.
  3. Se nenhuma revisitação for encontrada, retorna `False`.

#### Inicializar Indivíduo

```python
def inicializar_individuo(posicao_inicial: int) -> List[int]:
    """
    Inicializa um indivíduo com um caminho válido começando na posição inicial.

    Tenta construir um caminho até que não seja mais possível adicionar movimentos válidos.
    """
    caminho = [posicao_inicial]
    while True:
        movimentos = obter_movimentos_possiveis(caminho[-1])
        movimentos_possiveis = [m for m in movimentos if m not in caminho]
        if not movimentos_possiveis:
            break
        proximo_movimento = random.choice(movimentos_possiveis)
        caminho.append(proximo_movimento)
    return caminho
```

**Descrição:**

- **Objetivo:** Criar um caminho inicial para um indivíduo, começando de uma posição inicial aleatória.

- **Processo:**
  1. Inicia o caminho com a posição inicial.
  2. Repetidamente adiciona um movimento válido que não foi visitado anteriormente.
  3. Para quando não há mais movimentos válidos disponíveis.

#### Inicializar População

```python
def inicializar_populacao(tamanho: int = POPULACAO_INICIAL_TAMANHO) -> List[List[int]]:
    """
    Inicializa a população com indivíduos gerados aleatoriamente a partir de posições iniciais aleatórias.
    """
    populacao = []
    for _ in range(tamanho):
        pos_inicial = random.randint(0, TAMANHO_TABULEIRO - 1)
        individuo = inicializar_individuo(pos_inicial)
        # Verifica se há revisitações
        if verificar_revisitas(individuo):
            print(f"[Aviso] Indivíduo inicializado com revisitação na posição {revisitas_registradas[-1]}.")
        populacao.append(individuo)
    return populacao
```

**Descrição:**

- **Objetivo:** Gerar a população inicial do AG.

- **Processo:**
  1. Para cada indivíduo na população:
     - Seleciona uma posição inicial aleatória.
     - Cria um caminho válido a partir dessa posição.
     - Verifica se há revisitações no caminho.
     - Adiciona o indivíduo à população.

#### Selecionar Elite

```python
def selecionar_elite(populacao: List[List[int]], fitness_func: Callable[[List[int]], int], elite_percentual: float = ELITE_PERCENTUAL) -> List[List[int]]:
    """
    Seleciona a elite da população baseada na aptidão.
    """
    populacao_ordenada = sorted(populacao, key=lambda ind: fitness_func(ind), reverse=True)
    elite_tamanho = max(1, int(len(populacao) * elite_percentual))
    elite = populacao_ordenada[:elite_tamanho]
    
    # Verifica se a elite possui revisitações
    for ind in elite:
        if verificar_revisitas(ind):
            print(f"[Erro] Elite contém indivíduo com revisitação na posição {revisitas_registradas[-1]}.")
    
    return elite
```

**Descrição:**

- **Objetivo:** Selecionar os melhores indivíduos (elite) da população atual para reprodução.

- **Processo:**
  1. Ordena a população com base na aptidão (fitness).
  2. Seleciona os top `elite_percentual` indivíduos como elite.
  3. Verifica se algum indivíduo da elite possui revisitações.

#### Crossover

```python
def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Realiza o crossover entre dois pais para gerar dois filhos.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1.copy(), parent2.copy()

    ponto_corte = random.randint(1, min(len(parent1), len(parent2)) - 1)

    # Filho 1
    filho1 = parent1[:ponto_corte]
    for move in parent2:
        if move not in filho1:
            filho1.append(move)
            if len(filho1) == TAMANHO_TABULEIRO:
                break

    # Filho 2
    filho2 = parent2[:ponto_corte]
    for move in parent1:
        if move not in filho2:
            filho2.append(move)
            if len(filho2) == TAMANHO_TABULEIRO:
                break

    # Verifica se os filhos possuem revisitações
    if verificar_revisitas(filho1):
        print(f"[Aviso] Filho1 criado com revisitação na posição {revisitas_registradas[-1]}.")
    if verificar_revisitas(filho2):
        print(f"[Aviso] Filho2 criado com revisitação na posição {revisitas_registradas[-1]}.")

    return filho1, filho2
```

**Descrição:**

- **Objetivo:** Combinar duas soluções (pais) para produzir novas soluções (filhos).

- **Processo:**
  1. Seleciona um ponto de corte aleatório comum aos dois pais.
  2. Gera dois filhos combinando as partes dos pais:
     - **Filho 1:** Parte inicial de `parent1` + movimentos exclusivos de `parent2`.
     - **Filho 2:** Parte inicial de `parent2` + movimentos exclusivos de `parent1`.
  3. Garante que os filhos não revisitem posições já visitadas.
  4. Registra qualquer revisitação detectada nos filhos.

#### Mutação

```python
def mutacao(individual: List[int]) -> List[int]:
    """
    Realiza a mutação em um indivíduo.
    """
    if len(individual) < 2:
        return individual.copy()

    # Seleciona um ponto de mutação, excluindo a posição inicial
    ponto_mutacao = random.randint(1, len(individual) - 1)

    pos_atual = individual[ponto_mutacao - 1]
    movimentos = obter_movimentos_possiveis(pos_atual)

    # Exclui movimentos que já foram visitados antes do ponto de mutacao
    movimentos_possiveis = [m for m in movimentos if m not in individual[:ponto_mutacao]]

    if movimentos_possiveis:
        novo_movimento = random.choice(movimentos_possiveis)
        # Substitui o movimento no ponto de mutacao
        novo_individual = individual[:ponto_mutacao] + [novo_movimento]
        # Continua a construir o caminho a partir do novo movimento
        while True:
            movimentos = obter_movimentos_possiveis(novo_individual[-1])
            movimentos_possiveis = [m for m in movimentos if m not in novo_individual]
            if not movimentos_possiveis:
                break
            proximo_movimento = random.choice(movimentos_possiveis)
            novo_individual.append(proximo_movimento)
        # Verifica se a mutação resultou em revisitações
        if verificar_revisitas(novo_individual):
            print(f"[Aviso] Mutação resultou em revisitação na posição {revisitas_registradas[-1]}.")
        return novo_individual
    else:
        return individual.copy()
```

**Descrição:**

- **Objetivo:** Introduzir variação nos indivíduos para explorar novas soluções.

- **Processo:**
  1. Seleciona um ponto aleatório na sequência de movimentos (exceto o inicial).
  2. Escolhe um novo movimento válido a partir da posição anterior ao ponto de mutação.
  3. Substitui o movimento no ponto de mutação e continua a construir o caminho a partir daí.
  4. Garante que não haja revisitações após a mutação.
  5. Registra qualquer revisitação detectada.

#### Algoritmo Genético (AG)

```python
def ag(pop: List[List[int]], fitness_func: Callable[[List[int]], int], geracoes: int = NUM_GENERACOES) -> List[int]:
    """
    Implementa o Algoritmo Genético com crossover e mutação.

    Parameters
    ----------
    pop : List[List[int]]
        População inicial de indivíduos.
    fitness_func : Callable[[List[int]], int]
        Função de avaliação da aptidão.
    geracoes : int
        Número de gerações para executar o algoritmo.

    Returns
    -------
    List[int]
        O melhor indivíduo encontrado após todas as gerações.
    """
    for geracao in range(geracoes):
        # Avalia a aptidão de todos os indivíduos
        populacao_com_fitness = [(ind, fitness_func(ind)) for ind in pop]
        populacao_com_fitness.sort(key=lambda x: x[1], reverse=True)
        melhor_individuo, melhor_fitness = populacao_com_fitness[0]

        # Exibe o progresso a cada 500 gerações
        if geracao % 500 == 0 or geracao == geracoes - 1:
            print(f"Geração {geracao}: Melhor Aptidão = {melhor_fitness}")

        # Verifica se encontrou uma solução perfeita
        if melhor_fitness == TAMANHO_TABULEIRO:
            print(f"Solução perfeita encontrada na geração {geracao}!")
            return melhor_individuo

        # Seleciona a elite
        elite = selecionar_elite(pop, fitness_func)

        # Gera a nova população
        nova_populacao = elite.copy()

        # Quantidade de crossover e mutação
        num_crossover = int((POPULACAO_INICIAL_TAMANHO - len(nova_populacao)) * TAXA_CROSSOVER)
        num_mutacao = int((POPULACAO_INICIAL_TAMANHO - len(nova_populacao)) * TAXA_MUTACAO)

        # Realiza crossover
        for _ in range(num_crossover // 2):
            if len(elite) < 2:
                break
            parent1, parent2 = random.sample(elite, 2)
            filho1, filho2 = crossover(parent1, parent2)
            nova_populacao.append(filho1)
            nova_populacao.append(filho2)
            if len(nova_populacao) >= POPULACAO_INICIAL_TAMANHO:
                break

        # Realiza mutação
        for _ in range(num_mutacao):
            individuo = random.choice(elite)
            novo_individuo = mutacao(individuo)
            nova_populacao.append(novo_individuo)
            if len(nova_populacao) >= POPULACAO_INICIAL_TAMANHO:
                break

        # Preenche o restante da população com cópias da elite (sem alteração)
        while len(nova_populacao) < POPULACAO_INICIAL_TAMANHO:
            nova_populacao.append(random.choice(elite).copy())

        # Atualiza a população para a próxima geração
        pop = nova_populacao

    # Após todas as gerações, retorna o melhor indivíduo encontrado
    melhor_individuo = max(pop, key=lambda ind: fitness_func(ind))
    return melhor_individuo
```

**Descrição:**

- **Objetivo:** Evoluir a população de indivíduos através de seleção, crossover e mutação para encontrar uma solução completa para o Passeio do Cavalo.

- **Processo:**
  1. **Avaliação:** Calcula a aptidão de todos os indivíduos na população.
  2. **Seleção:** Ordena a população com base na aptidão e seleciona a elite.
  3. **Crossover e Mutação:** Gera novos indivíduos combinando e alterando os da elite.
  4. **Elitismo:** Mantém a elite intacta na nova população.
  5. **Repetição:** Repete o processo para o número definido de gerações ou até encontrar uma solução completa.
  6. **Retorno:** Retorna o melhor indivíduo encontrado após as gerações.

#### Visualização do Tabuleiro

```python
def tabuleiro(
    posicoes: List[int], possiveis_movimentos: Optional[List[int]] = None
) -> str:
    """
    Gera uma representação visual do tabuleiro com o caminho do cavalo.

    Parameters
    ----------
    posicoes : List[int]
        Sequência de posições visitadas pelo cavalo.
    possiveis_movimentos : list[int], optional
        Últimos movimentos possíveis do cavalo (não usado aqui).

    Returns
    -------
    str
        Representação visual do tabuleiro.
    """
    lado: int = int(np.sqrt(TAMANHO_TABULEIRO))
    if lado * lado != TAMANHO_TABULEIRO:
        raise ValueError(
            "O tamanho do tabuleiro deve ser um quadrado perfeito (e.g., 64 para 8x8)."
        )

    # Define etiquetas de colunas (A, B, C, ...)
    etiquetas_colunas: List[str] = [chr(ord("A") + i) for i in range(lado)]
    cabecalho: str = "   " + "  ".join(etiquetas_colunas)

    # Mapeia cada posição para o número da ordem em que foi visitada
    ordem_visita = {pos: idx + 1 for idx, pos in enumerate(posicoes)}

    linhas_visual: List[str] = [cabecalho]
    for linha in range(lado):
        etiqueta_linha: str = str(lado - linha)
        linha_visual: List[str] = [f"{etiqueta_linha} "]
        for coluna in range(lado):
            indice_atual: int = (lado * (lado - 1 - linha)) + coluna

            if indice_atual in ordem_visita:
                # Representa a posição do cavalo com o número da ordem em verde
                casa = Fore.GREEN + f"{ordem_visita[indice_atual]:2}" + Style.RESET_ALL
            else:
                # Representa outras casas com seus números em branco
                casa = f"{indice_atual:2}"

            linha_visual.append(casa + " ")
        linhas_visual.append(" ".join(linha_visual).rstrip())

    return "\n".join(linhas_visual)
```

**Descrição:**

- **Objetivo:** Gerar uma representação visual do tabuleiro de xadrez, destacando as posições visitadas pelo cavalo e a ordem em que foram visitadas.

- **Processo:**
  1. Calcula as coordenadas 2D do tabuleiro a partir dos índices unidimensionais.
  2. Mapeia cada posição visitada para o número do passo em que foi visitada.
  3. Gera uma string representando o tabuleiro com cores para facilitar a visualização:
     - **Verde:** Indica a ordem de visita das casas.
     - **Branco:** Indica casas não visitadas.

#### Exibir Movimentos

```python
def exibir_movimento(posicoes: List[int]) -> None:
    """
    Exibe o tabuleiro com o caminho do cavalo passo a passo.

    Parameters
    ----------
    posicoes : List[int]
        Sequência de posições visitadas pelo cavalo.
    """
    ordem_visita = {pos: idx + 1 for idx, pos in enumerate(posicoes)}
    lado: int = int(np.sqrt(TAMANHO_TABULEIRO))

    etiquetas_colunas: List[str] = [chr(ord("A") + i) for i in range(lado)]
    cabecalho: str = "   " + "  ".join(etiquetas_colunas)

    for idx, pos in enumerate(posicoes):
        # Limpa a tela
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Passeio do Cavalo - Visualização\n")
        print(f"Passo Atual: {idx + 1}\n")
        print(cabecalho)

        for linha in range(lado):
            etiqueta_linha: str = str(lado - linha)
            linha_visual: List[str] = [f"{etiqueta_linha} "]
            for coluna in range(lado):
                indice_atual: int = (lado * (lado - 1 - linha)) + coluna

                if indice_atual == pos:
                    # Representa a posição atual do cavalo com 'K' em vermelho
                    casa = Fore.RED + " K" + Style.RESET_ALL
                elif indice_atual in ordem_visita:
                    # Representa casas já visitadas com o número da ordem em azul
                    casa = Fore.BLUE + f"{ordem_visita[indice_atual]:2}" + Style.RESET_ALL
                else:
                    # Representa outras casas com seus números em branco
                    casa = f"{indice_atual:2}"

                linha_visual.append(casa + " ")
            print(" ".join(linha_visual).rstrip())

        # Delay para visualizar o movimento
        time.sleep(0.05)  # Ajuste o delay conforme necessário
```

**Descrição:**

- **Objetivo:** Mostrar uma animação passo a passo do movimento do cavalo pelo tabuleiro.

- **Processo:**
  1. Itera sobre cada posição visitada.
  2. Para cada posição:
     - Limpa a tela.
     - Exibe o passo atual.
     - Mostra o tabuleiro com a posição atual destacada em vermelho ('K') e as posições já visitadas em azul com o número do passo.
  3. Adiciona um pequeno atraso (`time.sleep(0.05)`) para simular a animação.

#### Execução do Algoritmo

```python
def main() -> None:
    print("Passeio do Cavalo - Algoritmo Genético com Crossover e Mutação\n")

    # Inicializa a população
    populacao_inicial = inicializar_populacao()

    # Executa o algoritmo genético
    melhor_individuo = ag(populacao_inicial, fitness, geracoes=NUM_GENERACOES)

    # Verifica se uma solução completa foi encontrada
    if fitness(melhor_individuo) == TAMANHO_TABULEIRO:
        print("\nMelhor Caminho Encontrado (Solução Completa):")
    else:
        print("\nMelhor Caminho Encontrado (Não Completo):")
    print(f"Aptidão: {fitness(melhor_individuo)}")

    # Exibe o caminho passo a passo
    exibir_movimento(melhor_individuo)

    # Exibe a representação final do tabuleiro
    print("\nRepresentação Final do Tabuleiro:")
    print(tabuleiro(melhor_individuo))

    # Exibe a lista ordenada de casas visitadas
    ordem_visita = {pos: idx + 1 for idx, pos in enumerate(melhor_individuo)}
    casas_visitadas_ordenadas = sorted(ordem_visita.items(), key=lambda x: x[1])
    lista_casas_visitadas = [f"{pos} (Passo {ord})" for pos, ord in casas_visitadas_ordenadas]
    
    print("\nLista Ordenada de Casas Visitadas:")
    print(", ".join(lista_casas_visitadas))

    # Verifica se todas as casas foram visitadas
    casas_visitadas = set(melhor_individuo)
    casas_faltantes = set(range(TAMANHO_TABULEIRO)) - casas_visitadas

    if not casas_faltantes:
        print("\nParabéns! O cavalo percorreu todas as 64 casas.")
    else:
        print(f"\nO cavalo não percorreu as seguintes casas ({len(casas_faltantes)} faltando):")
        print(", ".join(map(str, sorted(casas_faltantes))))

    # Verifica se houve revisitações
    if revisitas_registradas:
        print(f"\nRevisitações detectadas nas seguintes casas: {', '.join(map(str, revisitas_registradas))}")
    else:
        print("\nNenhuma revisitação detectada.")
```

**Descrição:**

- **Objetivo:** Executar todo o fluxo do Algoritmo Genético e exibir os resultados.

- **Processo:**
  1. **Inicialização:** Cria a população inicial.
  2. **Evolução:** Executa o AG por um número definido de gerações.
  3. **Resultados:**
     - Verifica se uma solução completa foi encontrada.
     - Exibe a visualização passo a passo dos movimentos.
     - Mostra a representação final do tabuleiro.
     - Lista as casas visitadas em ordem.
     - Indica se houve revisitações.

#### Limpar Tela

```python
def limpar_tela() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
```

**Descrição:**

- **Objetivo:** Limpar a tela do console para uma melhor visualização da animação.

---

## Análise e Resultados

### Convergência

A convergência do Algoritmo Genético para uma solução completa do Passeio do Cavalo depende de vários fatores:

- **Tamanho da População:** Populações maiores tendem a explorar melhor o espaço de soluções.
  
- **Taxas de Crossover e Mutação:** Taxas balanceadas promovem diversidade sem perder a qualidade das soluções.

- **Número de Gerações:** Mais gerações aumentam a chance de encontrar uma solução completa, mas também aumentam o tempo de execução.

- **Estrutura do Problema:** O Passeio do Cavalo é um problema complexo, e a eficácia do AG pode variar dependendo da implementação e dos parâmetros escolhidos.

### Eficiência

- **Tempo de Execução:** A eficiência do AG é influenciada pelo tamanho da população, número de gerações e complexidade das operações genéticas.
  
- **Recursos Computacionais:** Implementações otimizadas e o uso de bibliotecas eficientes (como NumPy) podem melhorar significativamente a performance.

### Revisitações

- **Prevenção de Revisitações:** Garantir que o cavalo não visite a mesma casa mais de uma vez é crucial para a validade da solução.
  
- **Registro de Revisitações:** O algoritmo registra quaisquer tentativas de revisitação, permitindo identificar e corrigir problemas na geração de indivíduos.

---

## Conclusão

Este documento detalhou a implementação de um **Algoritmo Genético (AG)** para resolver o **Passeio do Cavalo** no tabuleiro de xadrez. Abordamos os conceitos fundamentais dos AGs, adaptando-os para o contexto específico do problema, e detalhamos cada parte do código implementado.

Embora os AGs sejam poderosos para resolver problemas complexos, sua eficácia depende fortemente dos parâmetros escolhidos e das implementações dos operadores genéticos. No caso do Passeio do Cavalo, a implementação cuidadosa dos operadores de **crossover** e **mutação** é essencial para evitar revisitações e encontrar soluções completas.

**Recomendações Finais:**

- **Ajustar Parâmetros:** Testar diferentes tamanhos de população, taxas de crossover e mutação para otimizar os resultados.
  
- **Implementar Heurísticas:** Integrar heurísticas como a **Regra de Warnsdorff** pode guiar o AG de forma mais eficiente.
  
- **Melhorar a Diversidade:** Garantir uma população diversificada evita a convergência prematura para soluções subótimas.
  
- **Paralelização:** Considerar a execução paralela do AG para acelerar o processo de busca por soluções.

Com as melhorias contínuas e ajustes finos, o Algoritmo Genético pode se tornar uma ferramenta eficaz para resolver o Passeio do Cavalo e outros problemas complexos de otimização.

---

## Referências

1. **Mitchell, M.** (1998). *An Introduction to Genetic Algorithms*. MIT Press.
2. **Whitley, D.** (1994). *A Genetic Algorithm Tutorial*. Statistics and Computing, 4(2), 65-85.
3. **Warnsdorff's Rule:** Um método heurístico para resolver o Passeio do Cavalo, priorizando movimentos para casas com menos opções de saída.
4. **Wikipedia:** 
   - [Algoritmo Genético](https://pt.wikipedia.org/wiki/Algoritmo_genético)
   - [Passeio do Cavalo](https://pt.wikipedia.org/wiki/Passeio_do_cavalo)
5. **Python Documentation:** 
   - [random](https://docs.python.org/3/library/random.html)
   - [time](https://docs.python.org/3/library/time.html)
   - [os](https://docs.python.org/3/library/os.html)
6. **Colorama Documentation:** 
   - [Colorama](https://pypi.org/project/colorama/)

---

**Nota:** Este documento serve como material de estudo para entender a aplicação de Algoritmos Genéticos no problema do Passeio do Cavalo. Recomenda-se a execução e modificação do código apresentado para uma compreensão prática e aprofundada.
