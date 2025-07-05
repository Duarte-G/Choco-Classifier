import numpy as np
np.bool = bool

import random
import cv2
from pathlib import Path
import imgaug.augmenters as iaa

# Diretórios
pasta_entrada = Path("dataset_processado")
pasta_saida   = Path("dataset_aug")

# 1) Transformação logarítmica
def log_transform(image):
    img = image.astype(np.float32) / 255.0
    img = np.log1p(img)
    img = img / np.log1p(1.0) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

# 2) Transformação exponencial
def exp_transform(image):
    img = image.astype(np.float32) / 255.0
    img = np.expm1(img)
    img = img / np.expm1(1.0) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

# 3) Kernel de média 3×3
mean_kernel = np.ones((3, 3), dtype=np.float32) / 9.0

# Pipeline de augmentations
augmenter = iaa.Sequential([
    # 1) Log-transform
    iaa.Sometimes(
        0.7,
        iaa.Lambda(func_images=lambda images, random_state, parents, hooks: [log_transform(img) for img in images], name="LogTransform")
    ),
    # 2) Exp-transform
    iaa.Sometimes(
        0.7,
        iaa.Lambda(func_images=lambda images, random_state, parents, hooks: [exp_transform(img) for img in images], name="ExpTransform")
    ),
    # 3) Filtro da média 3×3
    iaa.Sometimes(
        0.7,
        iaa.Convolve(matrix=mean_kernel, name="MeanFilter")
    )
    # 4) Flip horizontal ou vertical (RESOLVIDO POSTERIORMENTE PRA FACILITAR SEGMENTACAO)
    # iaa.Sometimes(
    #     0.7,
    #     iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]), name="Flip"
    # ),
    # 5) Rotação aleatória (PROBLEMAS NA HORA DA SEGMENTACAO)
    # iaa.Sometimes(
    #     0.7,
    #     iaa.Affine(rotate=(-45, 45)), name="RandomRotate"
    # )
], random_order=True)

# Parâmetros
n_augs = 3

# Cria estrutura de saída
pasta_saida.mkdir(exist_ok=True)
for classe in sorted(pasta_entrada.iterdir()):
    if classe.is_dir():
        (pasta_saida / classe.name).mkdir(exist_ok=True)

# Processa cada imagem de cada classe
for classe in sorted(pasta_entrada.iterdir()):
    if not classe.is_dir():
        continue
    pasta_classe_in  = classe
    pasta_classe_out = pasta_saida / classe.name
    
    # lista de extensões suportadas
    for fname in sorted(pasta_classe_in.iterdir()):
        if not fname.suffix.lower() in (".png", ".jpg", ".jpeg"):
            continue
        
        # 1) Copia original
        dst_original = pasta_classe_out / fname.name
        if not dst_original.exists():
            dst_original.write_bytes(fname.read_bytes())
        
        # 2) Cria imagens aumentadas
        img = cv2.imread(str(fname))
        for i in range(1, n_augs + 1):
            aug = augmenter(image=img)

            # decide manualmente flip com probabilidade 0.7
            flip_tag = ""
            if random.random() < 0.7:
                if random.random() < 0.5:
                    aug = cv2.flip(aug, 1)
                    flip_tag = "_flipH"
                else:
                    aug = cv2.flip(aug, 0)
                    flip_tag = "_flipV"

            novo_nome = f"{fname.stem}_aug{i}{flip_tag}{fname.suffix}"
            cv2.imwrite(str(pasta_classe_out / novo_nome), aug)

print(f"Processamento concluído em: {pasta_saida.resolve()}")
