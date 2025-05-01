import os
from PIL import Image
import json

# Variaveis
pasta_entrada = "../dataset"     
pasta_saida = "../dataset_processado"     
image_size = (64, 64)                   # Tamanho das imagens final
center_crop = 1                         # 0 = comprime a imagem, 1 = prioriza o centro da imagem

# Criar diretório de saída se não existir
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Obter a lista de pastas (classes) em ordem alfabética
classes = sorted(os.listdir(pasta_entrada))
mapeamento = {indice: classe for indice, classe in enumerate(classes)}
mapeamento_reverso = {classe: indice for indice, classe in enumerate(classes)}

# Salvar o dicionário de mapeamento em formato JSON para uso posterior
with open(os.path.join(pasta_saida, "mapeamento.json"), "w") as f:
    json.dump(mapeamento, f, indent=4)

print("Mapeamento de classes:")
for indice, classe in mapeamento.items():
    print(f"{indice}: {classe}")

# Processar cada classe
for indice, classe in mapeamento.items():
    # Criar a pasta numerada correspondente no diretório de saída
    pasta_numerada = os.path.join(pasta_saida, str(indice))
    if not os.path.exists(pasta_numerada):
        os.makedirs(pasta_numerada)

    # Caminho para a pasta da classe original
    pasta_classe = os.path.join(pasta_entrada, classe)
    if not os.path.isdir(pasta_classe):
        continue

    # Processar imagens na pasta da classe
    for arquivo in os.listdir(pasta_classe):
        caminho_completo = os.path.join(pasta_classe, arquivo)
        if not (os.path.isfile(caminho_completo) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))):
            continue
        try:
            with Image.open(caminho_completo) as img:
                # Center crop
                if center_crop:
                    largura, altura = img.size
                    tamanho_minimo = min(largura, altura)
                    esquerda = (largura - tamanho_minimo) / 2
                    topo = (altura - tamanho_minimo) / 2
                    direita = esquerda + tamanho_minimo
                    inferior = topo + tamanho_minimo
                    img = img.crop((esquerda, topo, direita, inferior))
                
                # Redimensionar
                img_redimensionada = img.resize(image_size, Image.LANCZOS)

                # Salvar imagem processada na pasta numerada correspondente
                caminho_saida = os.path.join(pasta_numerada, arquivo)
                img_redimensionada.save(caminho_saida)

            print(f"Processado: {arquivo} -> Classe: {classe} -> Label: {indice}")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

print("\nProcessamento concluído!")
print(f"Dataset processado salvo em: {pasta_saida}")
print(f"Mapeamento de classes salvo em: {os.path.join(pasta_saida, 'mapeamento.json')}")