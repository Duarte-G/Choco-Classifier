import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os

# Arquivo CSV e diretório do mapeamento
arquivo_csv = "../dataset_csv/train.csv"
arquivo_mapeamento = "../dataset_processado/mapeamento.json"

# Verificar se o arquivo existe
if not os.path.exists(arquivo_csv):
    print(f"Erro: Arquivo {arquivo_csv} não encontrado.")
    exit()

# Carregar o mapeamento de classes (opcional, apenas para mostrar o nome da classe)
if os.path.exists(arquivo_mapeamento):
    with open(arquivo_mapeamento, "r") as f:
        mapeamento = json.load(f)
    mostrar_nome_classe = True
else:
    mostrar_nome_classe = False
    print("Arquivo de mapeamento não encontrado. Apenas o número da classe será exibido.")

# Carregar o arquivo CSV
print(f"Carregando o arquivo {arquivo_csv}...")
df = pd.read_csv(arquivo_csv)
print(f"Arquivo carregado. Total de {len(df)} imagens.")

# Selecionar uma amostra aleatória
indice_aleatorio = random.randint(0, len(df) - 1)
amostra = df.iloc[indice_aleatorio]

# Obter o rótulo e os pixels
rotulo = int(amostra['label'])
pixels = amostra.iloc[1:].values  # Todos os valores exceto o rótulo

# Calcular dimensões da imagem (assumindo que é quadrada)
tamanho_imagem = int(np.sqrt(len(pixels)))
print(f"Tamanho da imagem: {tamanho_imagem}x{tamanho_imagem} pixels")

# Remodelar os pixels para o formato da imagem
imagem = pixels.reshape(tamanho_imagem, tamanho_imagem)

# Mostrar a imagem
plt.figure(figsize=(8, 8))

# Título com o rótulo e o nome da classe (se disponível)
if mostrar_nome_classe and str(rotulo) in mapeamento:
    nome_classe = mapeamento[str(rotulo)]
    titulo = f"Classe: {rotulo} - {nome_classe}"
else:
    titulo = f"Classe: {rotulo}"

plt.title(titulo, fontsize=16)
plt.imshow(imagem, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"Imagem reconstruída da classe {rotulo}")