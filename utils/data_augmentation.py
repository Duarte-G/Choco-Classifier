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
    ),
    # 1) 
    iaa.Sometimes(
        0.2, 
        iaa.Lambda(
            func_images=lambda imgs, random_state, parents, hooks: [
                np.clip(
                    (np.log1p(img.astype(np.float32)/255.0 * 5) / np.log1p(5.0) * 255.0),
                    0, 255
                ).astype(np.uint8)
                for img in imgs
            ],
            name="StrongLogTransform"
        )
    ),
    # 2) Exp-transform reforçado (fator 5×)
    iaa.Sometimes(
        0.2,
        iaa.Lambda(
            func_images=lambda imgs, random_state, parents, hooks: [
                np.clip(
                    (np.expm1(img.astype(np.float32)/255.0 * 2) / np.expm1(2.0) * 255.0),
                    0, 255
                ).astype(np.uint8)
                for img in imgs
            ],
            name="StrongExpTransform"
        )
    ),

    # 3) Filtro da média 5×5 aplicado duas vezes
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((2, 2), [  # força a aplicação de 2 cópias
            iaa.Convolve(matrix=np.ones((5,5), dtype=np.float32)/25),
            iaa.Convolve(matrix=np.ones((5,5), dtype=np.float32)/25),
        ]),
        name="StrongMeanFilter"
    ),
    # 2) Filtros extras (50% de chance cada)
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.5, 1.5)),      name="GaussianBlur"),
    iaa.Sometimes(0.5, iaa.MotionBlur(k=(5, 15)),             name="MotionBlur"),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(5, 20)), name="AddGaussianNoise"),
    iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0)),         name="ContrastAdjust"),
    iaa.Sometimes(0.5, iaa.Add((-30, 30), per_channel=0.5),    name="BrightnessShift"),
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
n_augs = 6

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
