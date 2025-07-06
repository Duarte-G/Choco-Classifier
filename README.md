<p align="center">
  <img src="https://github.com/user-attachments/assets/49bbf074-215a-470b-b81f-4078f2506894" alt="Capa" width="600" />
</p>

# Classificador de Bombons

Este projeto aplica técnicas de visão computacional e deep learning para identificar diferentes marcas de bombons (por exemplo, Alpino, Charge etc.) a partir de imagens das embalagens.

## Base de dados

A base de dados foi construida a partir da captura manual de 10 diferentes tipos de chocolate, sendo eles: Alpino, Amor carioca, Charge, Chokito, Galak, Lollo, Negresco, Prestigio, Sensação e Smash.

Os dados iniciais estão armazenados na pasta ``dataset_processado/``, no qual apenas foi normalizado para 224x224, futuras etapas estão presentes nas pastas ``dataset_aug``, ``dataset_segm`` e ``dataset_normalizado``.

O dataset processado apresenta 16 imagens por classe, sendo 8 com fundo branco e 8 com fundo preto, totalizando 160 imagens.

## Processamentos nas imagens

Foram realizados alguns ajustes nas imagens para facilitar futuras etapas, entre as principais estão:

* Aumento de dados: Foi gerado 6 imagens para cada original com diferentes configurações providas pelo imgaug.

* Segmentação das imagens: Foi gerado uma versão da ground truth para cada uma das imagens presentes.

* Anotação de imagens: Foi gerado as bounding boxes a partir das imagens segmentadas.

* Normalização dos dados: Foi gerado uma versão dos dados normalizados pela equalização do histograma.

## Modelos treinados

### Classificador (Random Forest)

O classificador clássico utiliza a Random Forest treinada a partir de vetores de características extraídas de cada bombom segmentado. Para cada recorte, foram calculadas estatísticas de cor (em BGR, HSV e LAB), textura (por filtros de Gabor e gradientes Sobel) e forma (área, perímetro, circularidade, razão de aspecto e solidez). Esses vetores padronizados foram normalizados e divididos em 80% de treino, 10% de validação e 10% de teste.

O modelo final apresenta uma boa detecção quando utilizado em cenários controlados e não muito complexos visualmente, o F1-Score resultante foi 0.79.

### CNN (Criando as camadas)

O classificador baseado em CNN aprende diretamente das imagens segmentadas de cada bombom, sem necessidade de extração manual de características. A arquitetura tem três blocos convolucionais (filtros 3×3, ReLU, pooling e dropout) e duas camadas densas no final. O conjunto foi dividido em 80% treino, 10% validação e 10% teste, o modelo foi treinado com o otimizador Adam (learning rate 1×10⁻⁴), função de perda sparse_categorical_crossentropy e monitorado com EarlyStopping e redução de learning rate.

O modelo final apresenta uma boa capacidade de detecção dos bombons, superando por pouco o classificador clássico em alguns casos, o F1‑Score resultante foi 0.8.
