import sys
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

# ==============================================================================
def getIrisData():
    X = []
    Y = []
    with open('iris.csv', 'r') as r:
        reader = csv.DictReader(r, delimiter=',')
        for row in reader:
            x = [float(row['Sepal.Length']), float(row['Petal.Length'])]
            if row['Species'] == 'setosa':
                y = 1
            else:
                y = 0
            X.append(x)
            Y.append(y)
    return X, Y, 'Flor de Íris'

# ==============================================================================
def getANDData():
    X = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
    Y = [ 0, 0, 0, 1 ]
    return X, Y, 'Operador E'

# ==============================================================================
def getORData():
    X = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
    Y = [ 0, 1, 1, 1 ]
    return X, Y, 'Operador OU'

# ==============================================================================
def main(args):

    # Obtém os dados para teste (escolha um método!)
    #X, Y, probName = getANDData()
    #X, Y, probName = getORData()
    X, Y, probName = getIrisData()

    # Cria o Perceptron de 2 características
    clf = Perceptron(2)

    # Treina o Perceptron
    clf.fit(X, Y)

    # Plota os dados
    pal = sns.color_palette('colorblind', 3)
    fig, ax = plt.subplots(1)
    X = np.array(X)
    xmin = min(X[:,0]) - 0.5
    xmax = max(X[:,0]) + 0.5
    ymin = min(X[:,1]) - 0.5
    ymax = max(X[:,1]) + 0.5
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.set_title('Problema: {}'.format(probName), fontsize=20)
    ax.set_xlabel('$x_1$', fontsize=20)
    h = ax.set_ylabel('$x_2$', fontsize=20, labelpad=20)
    h.set_rotation(0)

    # Plota os pontos de cada classe
    lgClass0 = None
    lgClass1 = None
    for x, y in zip(X, Y):
        if y == 0:
            color = pal[0]
        else:
            color = pal[1]
        obj, = ax.plot([x[0]], [x[1]], c=color, marker='o', markersize=8,
                                       linestyle='None')
        if y == 0:
            lgClass0 = obj
        else:
            lgClass1 = obj

    # Plota a fronteira de separação (isto é, o hiperplano definido pelos pesos)
    # Para isso, usa uma grade de valores previstos com o próprio classificador
    # treinado a partir dos dados

    h = 0.005 # Tamanho dos "passos" para a grade a ser criada
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

    Z = []
    for x in np.c_[xx.ravel(), yy.ravel()]:
        Z.append(clf.predict(x.tolist()))

    Z = np.array(Z).reshape(xx.shape)
    CS = ax.contour(xx, yy, Z, colors=[pal[2]])
    #ax.axis('off')

    label = 'Hiperplano separador ($w = [{:.2f}, {:.2f}]$, $b = {:.2f}$)'
    ax.legend([lgClass0, lgClass1, CS.collections[0]],
            ['Classe A', 'Classe B', label.format(clf.w[0], clf.w[1], clf.w[2])],
            ncol=3, loc='upper right', prop={'size': 12})

    plt.show()

# ==============================================================================
class Perceptron:
    '''
    Classe que implementa o classificador linear Perceptron.
    '''

    # --------------------------------------------------------------------------
    def __init__(self, n):
        '''
        Construtor da classe.

        Parâmetros
        ----------
        n: int
            Número de características do problema.
        '''
        # Inicializa a lista de pesos
        self.w = [0 for _ in range(n+1)]

    # --------------------------------------------------------------------------
    def heaviside(self, net):
        '''
        Função de ativação degrau (heavside).
        Sobre: https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_de_Heaviside)

        Parâmetros
        ----------
        net: double
            Valor "líquido" para verificação de ativação.

        Retornos
        --------
        act: int
            Valor de ativação: 0 se não ativado, 1 se ativado.
        '''
        if net > 0:
            return 1
        else:
            return 0

    # --------------------------------------------------------------------------
    def predict(self, x):
        '''
        Faz a previsão da classe do exemplo x.

        Parâmetros
        ----------
        x: list
            Vetor de características de um exemplo a ser classificado.

        Retornos:
        class: int
            Classe a qual o exemplo x pertence: 0 indica a classe A, 1 indica a
            classe B.
        '''

        # Adiciona o viés ao final
        x_ = x.copy()
        x_.append(1)

        # Calcula o produto escalar entre os vetores de pesos (w) e entrada (x)
        net = 1
        for i in range(len(self.w)):
            net += self.w[i] * x_[i]

        # Executa a função de ativação
        f = self.heaviside(net)

        # Ativa ou não o neurônio dependendo do resultado da função de ativação
        if f > 0:
            return 1
        else:
            return 0

    # --------------------------------------------------------------------------
    def fit(self, X, Y):
        '''
        Treina o Perceptron com base nos exemplos e classes de treinamento
        dados.

        Parâmetros
        ----------
        X: list
            Vetor de exemplos, em que cada exemplo x é um vetor de
            características. Ou seja, é uma lista de listas.
        Y: list
            Vetor de inteiros com as classes verdadeiras (definidas como 0 ou 1)
            às quais os respectivos exemplos em X pertencem.
        '''

        while True:

            adjusted = False

            # Processa cada exemplo
            for x, y in zip(X, Y):

                # Faz a predição da classe do exemplo atual
                y_pred = self.predict(x)

                # Se o Perceptron errou, ajusta os pesos proporcionalmente ao
                # erro obtido
                if y != y_pred:

                    # Adiciona o viés ao final
                    x_ = x.copy()
                    x_.append(1)

                    # Ajusta os pesos
                    for i in range(len(self.w)):
                        self.w[i] += (y - y_pred) * x_[i]
                    adjusted = True

            # Verifica se convergiu (se não houve ajuste, os pesos já estão
            # corretamente ajustados aos exemplos de treinamento). Nesse
            # caso, o treinamento acabou! :)
            if not adjusted:
                break

# ------------------------------------------------------------------------------
# namespace verification for invoking main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])