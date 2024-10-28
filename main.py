# cavalo_xadrez_ag_improved.py
import numpy as np
from typing import Optional, Callable, Tuple
from colorama import Fore, Style, init
import random
import time
import os

# Inicializa o Colorama
init(autoreset=True)

# CONSTANTES
POSSIVEIS_MOVIMENTOS_CAVALO: np.ndarray = np.array(
    [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)],
    dtype=np.int32,
)
LADO_TABULEIRO = 64
TAMANHO_TABULEIRO = LADO_TABULEIRO * LADO_TABULEIRO

# Parâmetros do Algoritmo Genético
POPULACAO_INICIAL_TAMANHO = 200  # Maior diversidade
NUM_GENERACOES = 20000
ELITE_PERCENTUAL = 0.2  # 20% de elitismo
TAXA_CROSSOVER = 0.8  # 80% de indivíduos sofrem crossover
TAXA_MUTACAO = 0.2  # 20% de indivíduos sofrem mutação

# Variável global para registrar revisitações
revisitas_registradas = []


def obter_movimentos_possiveis(
    pos_atual: int, tamanho: int = TAMANHO_TABULEIRO
) -> list[int]:
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


def fitness(individual: list[int]) -> int:
    """
    Avalia a aptidão de um indivíduo.

    A aptidão é definida como o número de posições únicas visitadas pelo cavalo.
    """
    return len(set(individual))


def verificar_revisitas(individual: list[int]) -> bool:
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


def inicializar_individuo(posicao_inicial: int) -> list[int]:
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


def inicializar_populacao(tamanho: int = POPULACAO_INICIAL_TAMANHO) -> list[list[int]]:
    """
    Inicializa a população com indivíduos gerados aleatoriamente a partir de posições iniciais aleatórias.
    """
    populacao = []
    for _ in range(tamanho):
        pos_inicial = random.randint(0, TAMANHO_TABULEIRO - 1)
        individuo = inicializar_individuo(pos_inicial)
        # Verifica se há revisitações
        if verificar_revisitas(individuo):
            print(
                f"[Aviso] Indivíduo inicializado com revisitação na posição {revisitas_registradas[-1]}."
            )
        populacao.append(individuo)
    return populacao


def selecionar_elite(
    populacao: list[list[int]],
    fitness_func: Callable[[list[int]], int],
    elite_percentual: float = ELITE_PERCENTUAL,
) -> list[list[int]]:
    """
    Seleciona a elite da população baseada na aptidão.
    """
    populacao_ordenada = sorted(
        populacao, key=lambda ind: fitness_func(ind), reverse=True
    )
    elite_tamanho = max(1, int(len(populacao) * elite_percentual))
    elite = populacao_ordenada[:elite_tamanho]

    # Verifica se a elite possui revisitações
    for ind in elite:
        if verificar_revisitas(ind):
            print(
                f"[Erro] Elite contém indivíduo com revisitação na posição {revisitas_registradas[-1]}."
            )

    return elite


def crossover(parent1: list[int], parent2: list[int]) -> Tuple[list[int], list[int]]:
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
        print(
            f"[Aviso] Filho1 criado com revisitação na posição {revisitas_registradas[-1]}."
        )
    if verificar_revisitas(filho2):
        print(
            f"[Aviso] Filho2 criado com revisitação na posição {revisitas_registradas[-1]}."
        )

    return filho1, filho2


def mutacao(individual: list[int]) -> list[int]:
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
    movimentos_possiveis = [
        m for m in movimentos if m not in individual[:ponto_mutacao]
    ]

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
            print(
                f"[Aviso] Mutação resultou em revisitação na posição {revisitas_registradas[-1]}."
            )
        return novo_individual
    else:
        return individual.copy()


def ag(
    pop: list[list[int]],
    fitness_func: Callable[[list[int]], int],
    geracoes: int = NUM_GENERACOES,
) -> list[int]:
    """
    Implementa o Algoritmo Genético com crossover e mutação.

    Parameters
    ----------
    pop : list[list[int]]
        População inicial de indivíduos.
    fitness_func : Callable[[list[int]], int]
        Função de avaliação da aptidão.
    geracoes : int
        Número de gerações para executar o algoritmo.

    Returns
    -------
    list[int]
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
        num_crossover = int(
            (POPULACAO_INICIAL_TAMANHO - len(nova_populacao)) * TAXA_CROSSOVER
        )
        num_mutacao = int(
            (POPULACAO_INICIAL_TAMANHO - len(nova_populacao)) * TAXA_MUTACAO
        )

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


def tabuleiro(
    posicoes: list[int], possiveis_movimentos: Optional[list[int]] = None
) -> str:
    """
    Gera uma representação visual do tabuleiro com o caminho do cavalo.

    Parameters
    ----------
    posicoes : list[int]
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
    etiquetas_colunas: list[str] = [chr(ord("A") + i) for i in range(lado)]
    cabecalho: str = "   " + "  ".join(etiquetas_colunas)

    # Mapeia cada posição para o número da ordem em que foi visitada
    ordem_visita = {pos: idx + 1 for idx, pos in enumerate(posicoes)}

    linhas_visual: list[str] = [cabecalho]
    for linha in range(lado):
        etiqueta_linha: str = str(lado - linha)
        linha_visual: list[str] = [f"{etiqueta_linha} "]
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


def exibir_movimento(posicoes: list[int]) -> None:
    """
    Exibe o tabuleiro com o caminho do cavalo passo a passo.

    Parameters
    ----------
    posicoes : list[int]
        Sequência de posições visitadas pelo cavalo.
    """
    ordem_visita = {pos: idx + 1 for idx, pos in enumerate(posicoes)}
    lado: int = int(np.sqrt(TAMANHO_TABULEIRO))

    etiquetas_colunas: list[str] = [chr(ord("A") + i) for i in range(lado)]
    cabecalho: str = "   " + "  ".join(etiquetas_colunas)

    for idx, pos in enumerate(posicoes):
        # Limpa a tela
        os.system("cls" if os.name == "nt" else "clear")
        print("Passeio do Cavalo - Visualização\n")
        print(f"Passo Atual: {idx + 1}\n")
        print(cabecalho)

        for linha in range(lado):
            etiqueta_linha: str = str(lado - linha)
            linha_visual: list[str] = [f"{etiqueta_linha} "]
            for coluna in range(lado):
                indice_atual: int = (lado * (lado - 1 - linha)) + coluna

                if indice_atual == pos:
                    # Representa a posição atual do cavalo com 'K' em vermelho
                    casa = Fore.RED + " K" + Style.RESET_ALL
                elif indice_atual in ordem_visita:
                    # Representa casas já visitadas com o número da ordem em azul
                    casa = (
                        Fore.BLUE + f"{ordem_visita[indice_atual]:2}" + Style.RESET_ALL
                    )
                else:
                    # Representa outras casas com seus números em branco
                    casa = f"{indice_atual:2}"

                linha_visual.append(casa + " ")
            print(" ".join(linha_visual).rstrip())

        # Delay para visualizar o movimento
        time.sleep(0.05)  # Ajuste o delay conforme necessário


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
    lista_casas_visitadas = [
        f"{pos} (Passo {ord})" for pos, ord in casas_visitadas_ordenadas
    ]

    print("\nLista Ordenada de Casas Visitadas:")
    print(", ".join(lista_casas_visitadas))

    # Verifica se todas as casas foram visitadas
    casas_visitadas = set(melhor_individuo)
    casas_faltantes = set(range(TAMANHO_TABULEIRO)) - casas_visitadas

    if not casas_faltantes:
        print(f"\nO cavalo percorreu todas as {TAMANHO_TABULEIRO} casas.")
    else:
        print(
            f"\nO cavalo não percorreu as seguintes casas ({len(casas_faltantes)} faltando):"
        )
        print(", ".join(map(str, sorted(casas_faltantes))))

    # Verifica se houve revisitações
    if revisitas_registradas:
        print(
            f"\nRevisitações detectadas nas seguintes casas: {', '.join(map(str, revisitas_registradas))}"
        )
    else:
        print("\nNenhuma revisitação detectada.")


def limpar_tela() -> None:
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
