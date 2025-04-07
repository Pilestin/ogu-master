import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def apply_leaf_filters(img):
    """
    Görüntüyü alır ve filtreleri uygular, sonuçları döndürür
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply different filters
    # 1. Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    
    # 2. Edge detection
    edges = cv2.Canny(img, 100, 200)
    
    # 3. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes = list(cv2.split(lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    clahe_img = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)
    
    # 4. Use HSV color space to isolate potential diseased areas
    # Extract the saturation channel which can help highlight disease spots
    s_channel = hsv[:,:,1]
    
    # 5. Apply threshold to create binary image that highlights potential spots
    _, thresh = cv2.threshold(s_channel, 120, 255, cv2.THRESH_BINARY)
    
    return img_rgb, edges, clahe_img, s_channel, thresh, blur

def save_processed_image(image_path, output_folder, class_name):
    """
    Bir görüntüyü işler ve sonuçları kaydeder
    """
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Resim okunamadı: {image_path}")
        return
    
    # Filtreleri uygula
    img_rgb, edges, clahe_img, s_channel, thresh, blur = apply_leaf_filters(img)
    
    # Sonuç klasörünü hazırla
    class_output_dir = os.path.join(output_folder, class_name)
    if not os.path.exists(class_output_dir):
        os.makedirs(class_output_dir)
    
    # Dosya adını al
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Sonuçları birleştirip büyük bir resim oluştur
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Orijinal Görüntü")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Kenar Tespiti (Edge Detection)")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(clahe_img)
    plt.title("Kontrast Artırılmış (CLAHE)")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(s_channel, cmap='gray')
    plt.title("Doygunluk Kanalı (Saturation)")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(thresh, cmap='gray')
    plt.title("Eşikleme (Thresholding)")
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.imshow(blur)
    plt.title("Bulanıklaştırma (Gaussian Blur)")
    plt.axis("off")
    
    plt.suptitle(f"Yaprak Filtreleri: {os.path.basename(image_path)}", fontsize=16)
    plt.tight_layout()
    
    # Birleştirilmiş görüntüyü kaydet
    output_path = os.path.join(class_output_dir, f"{base_filename}_processed.png")
    plt.savefig(output_path)
    plt.close()  # Belleği temizle
    
    # Ayrıca her bir filtre sonucunu ayrı resim olarak da kaydedebiliriz
    # Bu kısmı kapatıldı - sadece birleşik resim kaydedilecek
    """
    # Edge görüntüsünü kaydet
    edge_path = os.path.join(class_output_dir, f"{base_filename}_edge.png")
    cv2.imwrite(edge_path, edges)
    
    # CLAHE görüntüsünü kaydet
    clahe_path = os.path.join(class_output_dir, f"{base_filename}_clahe.png")
    plt.imsave(clahe_path, clahe_img)
    
    # Threshold görüntüsünü kaydet
    thresh_path = os.path.join(class_output_dir, f"{base_filename}_thresh.png")
    cv2.imwrite(thresh_path, thresh)
    """

def process_all_images():
    """
    'leaves' klasöründeki tüm görüntüleri işler ve sonuçları 'processed_images' 
    klasöründe ilgili alt klasörlere kaydeder
    """
    # Girdi ve çıktı klasörleri
    input_dir = 'leaves'
    output_dir = 'processed_images'
    
    # Çıktı ana klasörünü oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu")
    
    # Sınıf klasörlerini listele
    try:
        class_folders = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Hata: '{input_dir}' klasörü bulunamadı!")
        return
    
    total_images = 0
    
    # Her sınıf için işlem yap
    for class_folder in class_folders:
        class_path = os.path.join(input_dir, class_folder)
        
        # Klasör değilse atla
        if not os.path.isdir(class_path):
            continue
        
        # Sınıf klasöründeki tüm dosyaları al
        files = [f for f in os.listdir(class_path) 
                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        
        if not files:
            print(f"'{class_folder}' klasöründe resim dosyası bulunamadı, atlanıyor.")
            continue
        
        print(f"\n'{class_folder}' sınıfında {len(files)} resim işleniyor...")
        
        # Her bir resmi işle
        for file in tqdm(files, desc=class_folder):
            image_path = os.path.join(class_path, file)
            save_processed_image(image_path, output_dir, class_folder)
            total_images += 1
    
    print(f"\nİşlem tamamlandı! Toplam {total_images} resim işlendi.")
    print(f"İşlenmiş resimler '{output_dir}' klasörünün ilgili alt klasörlerine kaydedildi.")

if __name__ == "__main__":
    process_all_images()
