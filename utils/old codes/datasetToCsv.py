import os
import numpy as np
import pandas as pd
from PIL import Image
import json
from sklearn.model_selection import train_test_split

# Diretório de entrada (dataset processado)
pasta_entrada = "../dataset_processado"
pasta_saida = "../dataset_csv"

# Criar diretório de saída se não existir
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Carregar o mapeamento de classes
with open(os.path.join(pasta_entrada, "mapeamento.json"), "r") as f:
    mapeamento = json.load(f)

print("Coletando todas as imagens...")
dados = []

# Percorrer todas as pastas numeradas
for label in sorted(os.listdir(pasta_entrada)):
    # Verificar se é uma pasta numérica
    if label.isdigit() and os.path.isdir(os.path.join(pasta_entrada, label)):
        label_int = int(label)
        pasta_label = os.path.join(pasta_entrada, label)
        
        # Processar todas as imagens da pasta
        for arquivo in os.listdir(pasta_label):
            caminho_completo = os.path.join(pasta_label, arquivo)
            
            # Verificar se é uma imagem
            if os.path.isfile(caminho_completo) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    # Carregar a imagem e converter para escala de cinza
                    imagem = Image.open(caminho_completo).convert('L')  # 'L' para escala de cinza
                    
                    # Converter imagem para array numpy e normalizar (0-255)
                    imagem_array = np.array(imagem).flatten()
                    
                    # Adicionar à lista de dados com o rótulo
                    dados.append((label_int, imagem_array))
                    
                except Exception as e:
                    print(f"Erro ao processar {arquivo}: {e}")

print(f"Total de imagens coletadas: {len(dados)}")

# Separar rótulos e características
labels = [item[0] for item in dados]
features = [item[1] for item in dados]

# Dividir os dados em conjuntos de treinamento, teste e validação
X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)    # TEST SIZE TALVEZ MEIO ALTO
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)     # TEST SIZE TALVEZ MEIO ALTO

# Função para salvar os dados em CSV
def salvar_csv(X, y, nome_arquivo):
    # Criar DataFrame
    df = pd.DataFrame(X)
    # Adicionar a coluna de rótulos no início
    df.insert(0, "label", y)
    
    # Salvar em CSV
    caminho_csv = os.path.join(pasta_saida, nome_arquivo)
    df.to_csv(caminho_csv, index=False)
    print(f"Arquivo {nome_arquivo} salvo com {len(df)} amostras")
    
    # Resumo de distribuição de classes
    classe_counts = pd.Series(y).value_counts().sort_index()
    print(f"Distribuição de classes em {nome_arquivo}:")
    for classe, contagem in classe_counts.items():
        nome_classe = mapeamento[str(classe)]
        print(f"  Classe {classe} ({nome_classe}): {contagem} amostras")

# Salvar os conjuntos em arquivos CSV
salvar_csv(X_train, y_train, "train.csv")
salvar_csv(X_test, y_test, "test.csv")
salvar_csv(X_val, y_val, "validation.csv")

print("\nProcessamento concluído!")
print(f"Arquivos CSV salvos em: {pasta_saida}")
print("Resumo da divisão dos dados:")
print(f"  Treinamento: {len(y_train)} amostras ({len(y_train)/len(labels)*100:.1f}%)")
print(f"  Teste: {len(y_test)} amostras ({len(y_test)/len(labels)*100:.1f}%)")
print(f"  Validação: {len(y_val)} amostras ({len(y_val)/len(labels)*100:.1f}%)")