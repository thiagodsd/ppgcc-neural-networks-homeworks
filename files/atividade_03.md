# 1 Descrição do Trabalho

Este trabalho tem como objetivo a implementação de uma rede neural do tipo Multi-Layer Perceptron (MLP) utilizando a biblioteca PyTorch. A implementação deve ser feita manualmente, sem o uso de implementações prontas, utilizando apenas componentes básicos do PyTorch como camadas, funções de ativação e otimizadores. O trabalho consiste em implementar e treinar uma rede neural MLP para resolver um problema de classificação. A implementação deve permitir a construção de uma arquitetura flexı́vel, com número variável de camadas e neurônios por camada.


## 1.1 Requisitos Funcionais

1. Implementação da MLP:
   * Implementar a MLP manualmente, usando a classe nn.Module do PyTorch.
   * A implementação deve permitir especificar a arquitetura da rede no formato [N1 , N2 , ..., Nk ], onde:
     * N1 representa o número de neurônios na camada de entrada
     * N2 , ..., Nk−1 representam o número de neurônios nas camadas escondidas
     * Nk representa o número de neurônios na camada de saı́da
     * Permitir configurar funções de ativação Tanh ou ReLU nas camadas escondidas.
2. Treinamento:
   * Implementar o treinamento da rede utilizando o otimizador SGD (Stochastic Gradient Descent).
   * Utilizar a função de perda CrossEntropy (nn.CrossEntropyLoss()) para problemas de classificação multiclasse e Binary Cross Entropy (nn.BCEWithLogitsLoss()) para binário.
   * Implementar a funcionalidade de validação durante o treinamento.
   * Implementar Early Stopping com Patience definido manualmente.
   * Implementar predição dos dados de teste com rede treinada.
   * Monitorar e registrar as métricas de desempenho (acurácia, perda) durante o treinamento.