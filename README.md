# 📝 Checkpoint 2 – Disruptive Architectures: IoT, IoB e Generative AI

## 👤 Aluno
**Nome:** Felipe Ulson Sora  
**RM:** 555462

Este repositório contém a implementação dos exercícios de **classificação multiclasse** e **regressão** usando **redes neurais em Keras** e modelos tradicionais do **scikit-learn**.  

---

## 🍷 Exercício 1 – Classificação Multiclasse (Wine Dataset)

### 🎯 Objetivo
Classificar vinhos em 3 classes diferentes com base em 13 atributos químicos.

### 🛠️ Modelos Utilizados
- **Rede Neural (Keras)**:
  - 2 camadas ocultas com 32 neurônios cada (ReLU)
  - Camada de saída com 3 neurônios (Softmax)
  - Perda: categorical_crossentropy
  - Otimizador: Adam
- **RandomForestClassifier (Scikit-learn)**
- **LogisticRegression (Scikit-learn)**

### ⚙️ Pré-processamento
- Normalização dos atributos com `StandardScaler`
- One-hot encoding das classes para a rede neural
- Divisão em treino/teste (80/20)

### 📊 Resultados
| Modelo                | Acurácia |
|-----------------------|----------|
| Rede Neural (Keras)   | 1.0000   |
| RandomForest          | 1.0000   |
| Logistic Regression   | 1.0000   |

### 📝 Discussão
- Todos os modelos atingiram acurácia perfeita no conjunto de teste, mostrando que o **Wine Dataset** possui atributos altamente discriminativos.  
- A **RandomForest** e a **Logistic Regression** são mais rápidas e simples, enquanto a **rede neural** é mais flexível e aplicável a problemas mais complexos.  
- Apesar da alta acurácia, recomenda-se validação cruzada em datasets pequenos para evitar overfitting e obter uma avaliação mais confiável.

---

## 🏠 Exercício 2 – Regressão (California Housing Dataset)

### 🎯 Objetivo
Prever o valor médio das casas na Califórnia com base em 8 atributos.

### 🛠️ Modelos Utilizados
- **Rede Neural (Keras)**:
  - 3 camadas ocultas: 64, 32 e 16 neurônios (ReLU)
  - Camada de saída com 1 neurônio (Linear)
  - Perda: MSE
  - Otimizador: Adam
- **LinearRegression (Scikit-learn)**
- **RandomForestRegressor (Scikit-learn)**

### ⚙️ Pré-processamento
- Normalização dos atributos com `StandardScaler`
- Divisão em treino/teste (80/20)

### 📊 Resultados
| Modelo                | MSE     | MAE    |
|-----------------------|---------|--------|
| Rede Neural (Keras)   | 0.27    | 0.34   |
| Linear Regression     | 0.50    | 0.54   |
| RandomForest Regressor| 0.26    | 0.33   |

### 📝 Discussão
- O **RandomForest Regressor** apresentou o melhor desempenho, com os menores valores de erro.  
- A **Rede Neural** conseguiu resultados próximos, mas poderia ser otimizada com mais épocas, regularização ou ajuste de hiperparâmetros.  
- A **Linear Regression** teve desempenho inferior, devido à sua limitação em capturar relações não lineares entre variáveis.  
- Conclusão: Para dados tabulares estruturados, **modelos baseados em árvores** (RandomForest) costumam superar redes neurais simples.

---

## 💡 Observações Finais
- Foram aplicadas técnicas de **normalização e encoding** para melhorar o desempenho dos modelos.  
- O repositório serve como referência de comparação entre **modelos de redes neurais** e **modelos tradicionais do scikit-learn** em problemas de **classificação e regressão**.
