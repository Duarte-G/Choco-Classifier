import re
from pathlib import Path
import shutil
import cv2

pasta_aug  = Path("dataset_aug")
pasta_segm = Path("dataset_segm")

for class_dir in sorted(pasta_aug.iterdir()):
    if not class_dir.is_dir():
        continue

    segm_class_dir = pasta_segm / class_dir.name
    segm_class_dir.mkdir(exist_ok=True)

    for img_path in sorted(class_dir.iterdir()):
        stem = img_path.stem
        
        # reconhece tanto _augX_flipH/V quanto apenas _augX
        m = re.match(r"(.+)_aug\d+(?:_(flipH|flipV))?$", stem)
        if not m:
            continue

        base_stem, flip_type = m.group(1), m.group(2)
        out_mask = segm_class_dir / f"{stem}_mask.png"
        if out_mask.exists():
            # já gerado, pula
            continue

        # caminho da máscara base
        base_mask_path = segm_class_dir / f"{base_stem}_mask.png"
        if not base_mask_path.exists():
            print(f"[!] Máscara base não encontrada para {stem}")
            continue

        if flip_type is None:
            # caso sem flip: apenas copia o arquivo
            shutil.copy2(base_mask_path, out_mask)
            print(f"Copiou máscara original para {stem} (sem flip)")
        else:
            # carrega e aplica flip na máscara
            mask = cv2.imread(str(base_mask_path), cv2.IMREAD_GRAYSCALE)
            if flip_type == "flipH":
                mask_flip = cv2.flip(mask, 1)
            else:  # flipV
                mask_flip = cv2.flip(mask, 0)
            cv2.imwrite(str(out_mask), mask_flip)
            print(f"Salvou máscara para {stem} ({flip_type})")

print("Concluído: máscaras para imagens com augX geradas em dataset_segm.")
