# 📝 Checkpoint 2 – Disruptive Architectures: IoT, IoB e Generative AI

## 👤 Aluno
**Nome:** Felipe Ulson Sora  
**RM:** 555462

## 💾 Descrição do Repositório

Este repositório contém as implementações dos exercícios desenvolvidos na disciplina **Disruptive Architectures: IoT, IoB e Generative AI**, divididos em duas partes:

- **Parte 01 – Redes Neurais**: aplicações de **classificação multiclasse** e **regressão** utilizando **redes neurais com Keras** e modelos tradicionais do **Scikit-learn**.

- **Parte 02 – Visão Computacional**: uso de **modelos pré-treinados** do **Hugging Face** e do **YOLOv8** para **classificação e detecção de objetos em imagens** (cães e gatos).

O projeto compara **diferentes abordagens de aprendizado de máquina e deep learning**, explorando desde **dados tabulares** até **imagens reais**.

---

## 🤖 Parte 01 - Redes Neurais

### 🍷 Exercício 1 – Classificação Multiclasse (Wine Dataset)

#### 🎯 Objetivo
Classificar vinhos em 3 classes diferentes com base em 13 atributos químicos.

#### 🛠️ Modelos Utilizados
- **Rede Neural (Keras)**:
  - 2 camadas ocultas com 32 neurônios cada (ReLU)
  - Camada de saída com 3 neurônios (Softmax)
  - Perda: categorical_crossentropy
  - Otimizador: Adam
- **RandomForestClassifier (Scikit-learn)**
- **LogisticRegression (Scikit-learn)**

#### ⚙️ Pré-processamento
- Normalização dos atributos com `StandardScaler`
- One-hot encoding das classes para a rede neural
- Divisão em treino/teste (80/20)

#### 📊 Resultados
| Modelo                | Acurácia |
|-----------------------|----------|
| Rede Neural (Keras)   | 1.0000   |
| RandomForest          | 1.0000   |
| Logistic Regression   | 1.0000   |

#### 📝 Discussão
- Todos os modelos atingiram acurácia perfeita no conjunto de teste, mostrando que o **Wine Dataset** possui atributos altamente discriminativos.  
- A **RandomForest** e a **Logistic Regression** são mais rápidas e simples, enquanto a **rede neural** é mais flexível e aplicável a problemas mais complexos.  
- Apesar da alta acurácia, recomenda-se validação cruzada em datasets pequenos para evitar overfitting e obter uma avaliação mais confiável.

### 🏠 Exercício 2 – Regressão (California Housing Dataset)

#### 🎯 Objetivo
Prever o valor médio das casas na Califórnia com base em 8 atributos.

#### 🛠️ Modelos Utilizados
- **Rede Neural (Keras)**:
  - 3 camadas ocultas: 64, 32 e 16 neurônios (ReLU)
  - Camada de saída com 1 neurônio (Linear)
  - Perda: MSE
  - Otimizador: Adam
- **LinearRegression (Scikit-learn)**
- **RandomForestRegressor (Scikit-learn)**

#### ⚙️ Pré-processamento
- Normalização dos atributos com `StandardScaler`
- Divisão em treino/teste (80/20)

#### 📊 Resultados
| Modelo                | MSE     | MAE    |
|-----------------------|---------|--------|
| Rede Neural (Keras)   | 0.27    | 0.34   |
| Linear Regression     | 0.50    | 0.54   |
| RandomForest Regressor| 0.26    | 0.33   |

#### 📝 Discussão
- O **RandomForest Regressor** apresentou o melhor desempenho, com os menores valores de erro.  
- A **Rede Neural** conseguiu resultados próximos, mas poderia ser otimizada com mais épocas, regularização ou ajuste de hiperparâmetros.  
- A **Linear Regression** teve desempenho inferior, devido à sua limitação em capturar relações não lineares entre variáveis.  
- Conclusão: Para dados tabulares estruturados, **modelos baseados em árvores** (RandomForest) costumam superar redes neurais simples.

### 💡 Observações Finais
- Foram aplicadas técnicas de **normalização e encoding** para melhorar o desempenho dos modelos.  
- O repositório serve como referência de comparação entre **modelos de redes neurais** e **modelos tradicionais do scikit-learn** em problemas de **classificação e regressão**.

--- 

## 🧠 Parte 02 - Visão Computacional

### 🎯 Objetivo

Explorar técnicas de **classificação e detecção de imagens** utilizando duas ferramentas de Visão Computacional distintas — **Hugging Face e YOLOv8** — para comparar seus resultados em imagens contendo **cães e gatos**.

### ⚙️ Ferramentas Utilizadas

#### 🧩 1. Hugging Face – Vision Transformer (ViT)

- Modelo: `google/vit-base-patch16-224`

- Tarefa: **Classificação de imagem**

- O modelo foi pré-treinado em milhões de imagens, sendo capaz de identificar a classe mais provável de um objeto em uma imagem completa.

- Utilizado por meio do `pipeline("image-classification")` da biblioteca `transformers`.

#### 🧠 2. YOLOv8 – Ultralytics

- Modelo: `yolov8n.pt` (versão leve pré-treinada no COCO Dataset)

- Tarefa: **Detecção de objetos**

- Identifica e localiza objetos dentro de uma imagem, desenhando caixas delimitadoras (bounding boxes) e mostrando o nível de confiança de cada detecção.

### 🐾 Conjunto de Imagens

Foram utilizadas três imagens contendo cães e gatos em diferentes contextos:

- `gato02.jpg` → apenas um gato

- `cachorros.jpg` → múltiplos cães

- `cachorro_gato.jpg` → cão e gato juntos

Essas imagens foram fornecidas em aula e representam um cenário ideal para comparar **classificação global (Hugging Face)** e **detecção localizada (YOLO)**.

### 🔍 Metodologia

O notebook executa as seguintes etapas:

**1. Instalação das bibliotecas**: `ultralytics`, `transformers`, `torch`, `pillow` e `opencv-python-headless`.

**2. Classificação (Hugging Face)**:
    - Cada imagem é processada por um modelo de Transformer visual (ViT) para prever a classe principal (ex: Egyptian cat, Labrador, etc.).

**3. Detecção (YOLOv8)**:
    - O modelo YOLOv8 é aplicado às mesmas imagens, retornando as classes detectadas (ex: cat, dog) e desenhando caixas delimitadoras.

**4. Visualização e Comparação**:
    - As imagens resultantes são exibidas diretamente no notebook com as detecções visuais.
    - Os resultados textuais de ambas as abordagens são exibidos lado a lado.

### 📊 Resultados Obtidos

| Imagem            | Hugging Face (Classificação) | YOLOv8 (Detecção)                       |
| ----------------- | ---------------------------- | --------------------------------------- |
| gato02.jpg        | *Egyptian cat* (0.25)        | 1 gato detectado (0.85)                 |
| cachorros.jpg     | *kelpie* (0.48)              | 3 cães detectados (0.91, 0.87, 0.77)    |
| cachorro_gato.jpg | *redbone* (0.78)             | 1 cão (0.76) e 1 gato (0.75) detectados |

### 📝 Discussão dos Resultados

- O **modelo do Hugging Face** foi eficaz em **classificar o tipo de animal predominante** na imagem, mas não distingue múltiplos objetos simultaneamente. Ele trata a imagem como um todo (visão global).

- O **YOLOv8**, por outro lado, conseguiu **detectar e localizar vários animais na mesma imagem**, mostrando sua força em **detecção de objetos**.
Ele fornece informações espaciais (posição e quantidade).

- As diferenças refletem dois paradigmas distintos da Visão Computacional:

    - **Classificação**: “O que há nesta imagem?”

    - **Detecção**: “O que há e onde está?”

- Em imagens simples (um único animal), ambos os modelos tiveram respostas coerentes.

- Em imagens com **múltiplos animais**, o **YOLOv8** se mostrou mais completo, enquanto o **Hugging Face** identificou apenas uma categoria dominante.

### 💡 Conclusão

- A combinação de **Hugging Face (ViT)** e **YOLOv8** permitiu comparar abordagens complementares da Visão Computacional.

- O **Hugging Face** se destaca pela **simplicidade e precisão em classificação geral**, enquanto o **YOLOv8** é mais poderoso para **detecção e análise em tempo real**.

- Esse experimento demonstra como diferentes arquiteturas de IA podem ser aplicadas a um mesmo problema, revelando a **diversidade de aplicações da Visão Computacional** em sistemas inteligentes.