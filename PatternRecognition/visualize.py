import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from datetime import datetime

def analyze_folder(folder_path, num_samples=5):
    files = sorted(os.listdir(folder_path))
    
    # Zaman damgasını parse etme
    timestamps = []
    for f in files:
        try:
            # Farklı formatlar için koşullu parsing
            if "USB" in f:
                dt_str = f.split('_')[-4] + '_' + f.split('_')[-3]
            else:
                dt_str = f.split('_')[1] + '_' + f.split('_')[2]
            dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
            timestamps.append(dt)
        except:
            print(f"Parsing error: {f}")
    
    # Zaman sıralaması
    sorted_files = [f for _,f in sorted(zip(timestamps, files))]
    
    # Görselleştirme
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        # İlk ve son 5 görsel
        for j, pos in enumerate(['early', 'late']):
            idx = i if pos == 'early' else -(i+1)
            img_path = os.path.join(folder_path, sorted_files[idx])
            
            ax = plt.subplot(2, num_samples, j*num_samples + i +1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Class {os.path.basename(folder_path)}\n{pos} stage\n{sorted_files[idx]}")
            plt.axis('off')
    
    # Zaman serisi analizi
    time_diffs = pd.Series(timestamps).diff().dt.total_seconds().dropna()
    print(f"\nClass {os.path.basename(folder_path)} Analysis:")
    print(f"Total images: {len(files)}")
    print(f"Time between shots: {time_diffs.mean():.1f} ± {time_diffs.std():.1f} sec")
    print(f"First image: {sorted_files[0]}")
    print(f"Last image: {sorted_files[-1]}")

# Tüm klasörler için analiz
for class_folder in sorted(os.listdir("leaves")):
    folder_path = os.path.join("leaves", class_folder)
    analyze_folder(folder_path)
    plt.tight_layout()
    plt.show()