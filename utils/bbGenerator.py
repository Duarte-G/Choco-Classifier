import os
import csv
import cv2
from pathlib import Path

# Diretórios
input_dir = Path("dataset_segm")
output_csv = Path("dataset_segm/bboxes.csv")

# Abre CSV para escrita
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x", "y", "w", "h"])
    
    # Percorre cada imagem de máscara
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for mask_path in sorted(class_dir.glob("*_mask.png")):
            # Lê máscara em tons de cinza
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Encontra contornos no objeto (foreground = 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # sem contorno: pula
                continue
            
            # Seleciona o contorno de maior área
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Grava no CSV (caminho relativo)
            rel_path = mask_path.relative_to(input_dir)
            writer.writerow([str(rel_path), x, y, w, h])

print(f"Bounding boxes escritas em: {output_csv.resolve()}")  
