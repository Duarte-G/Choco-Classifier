import os
import pandas as pd

DATASET_DIR = '../dataset'
OUTPUT_FILE = '../metadata/metadata.csv'

# Extração de metadados
def parse_filename(filename):
    base = os.path.basemname(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    class_id, seq, angle, background = parts
    return int(class_id), seq, angle, background

def generate_metadata():
    data = []
    for class_dir in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue
        for file in sorted(os.listdir(class_path)):
            if file.endswith('.png'):
                full_path = os.path.join(class_path, file)
                try:
                    class_id, seq, angle, background = parse_filename(file)
                    data.append({
                        'path': full_path,
                        'class': class_dir,
                        'class_id': class_id,
                        'sequence': seq,
                        'angle': angle,
                        'background': background
                    })
                except:
                    print(f"Erro ao processar arquivo: {file}")
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Metadados salvos em {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_metadata()