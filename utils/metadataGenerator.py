import os
import csv

pasta_entrada = "dataset_processado"
saida_csv = os.path.join(pasta_entrada, "metadata.csv")

with open(saida_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "class_name", "class_id", "image_id", "angle", "bg_color"])
    
    # percorre cada pasta de classe
    for class_name in sorted(os.listdir(pasta_entrada)):
        class_dir = os.path.join(pasta_entrada, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            # exemplo: 00-01-V1-B.png
            parts = fname.split("-")               # parts = ["00", "01", "V1", "B.png"]
            class_id = parts[0]
            image_id = parts[1]
            angle = parts[2]                       # "V1"
            bg_color = parts[3].split(".")[0]      # "B"
            # caminho relativo dentro do root
            rel_path = os.path.join(class_name, fname)
            writer.writerow([rel_path, class_name, class_id, image_id, angle, bg_color])

print(f"Arquivo de metadados salvo em: {saida_csv}")