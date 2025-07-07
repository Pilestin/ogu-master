# Bald Eagle Search (BES) & Golden Sine Algorithm (GSA) Benchmark Comparison

> Bu proje, **Algoritma Tasarımı ve Analizi** yüksek lisans dersi kapsamında gerçekleştirilmiştir. Meta-sezgisel optimizasyon algoritmalarından **Bald Eagle Search** ve **Golden Sine Algorithm** yöntemleri Python ile uygulanmış, temel test fonksiyonları üzerinde performansları karşılaştırılmıştır.

## 🔍 Proje Özeti

Bu çalışmada:

- Doğadan ilham alan iki optimizasyon algoritması uygulanmıştır:
  - 🦅 **Bald Eagle Search (BES)**: Kartalların avlanma davranışından esinlenir
  - 📐 **Golden Sine Algorithm (GSA)**: Altın oran ve sinüs dalgalanmasını kullanır
- 5 farklı benchmark fonksiyonu test edilmiştir:
  - Sphere
  - Rosenbrock
  - Ackley
  - Rastrigin
  - Griewank
- Her algoritma için performans metrikleri:
  - En iyi çözüm değeri
  - Çalışma süresi
  - Yakınsama hızı
- Görselleştirmeler:
  - Test fonksiyonlarının 3D yüzey grafikleri
  - Algoritmaların yakınsama grafikleri
  - Karşılaştırmalı performans tabloları

## 📊 Test Fonksiyonları

| Fonksiyon   | Tanım | Global Minimum |
|-------------|-------|----------------|
| Sphere      | \( f(x) = \sum x_i^2 \) | f(0,...,0) = 0 |
| Rosenbrock  | \( f(x) = \sum [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2] \) | f(1,...,1) = 0 |
| Ackley      | \( f(x) = -20e^{-0.2\sqrt{\frac{1}{n}\sum x_i^2}} - e^{\frac{1}{n}\sum \cos(2\pi x_i)} + 20 + e \) | f(0,...,0) = 0 |
| Rastrigin   | \( f(x) = 10n + \sum [x_i^2 - 10\cos(2\pi x_i)] \) | f(0,...,0) = 0 |
| Griewank    | \( f(x) = 1 + \sum \frac{x_i^2}{4000} - \prod \cos(\frac{x_i}{\sqrt{i}}) \) | f(0,...,0) = 0 |

## 🚀 Algoritmaların Teknik Özeti

### Bald Eagle Search (BES)
- **3 fazlı yapı**:
  1. **Alan Seçimi (Select Space)**: P = gbest + α·r·(mean - P)
  2. **Spiral Arama (Search in Space)**: Spiral hareket parametreleri (a_spiral, R_spiral)
  3. **Dalış (Swooping)**: Hiperbolik spiral ile hedef takibi
- **Parametreler**:
  - alpha: [1.5, 2.0] - Alan seçim katsayısı
  - a_spiral: [5, 10] - Spiral şekil parametresi
  - R_spiral: [0.5, 2.0] - Spiral yarıçap parametresi
  - c1, c2: [1, 2] - Dalış ağırlık katsayıları

### Golden Sine Algorithm (GSA)
- **Altın oran (τ ≈ 0.618) tabanlı arama**:
  - Pozisyon güncelleme: P = P·|sin(r₁)| - r₂·sin(r₁)·|x₁·gbest - x₂·P|
  - Altın kesit aralığı: [-π, π]
- **Avantajları**:
  - Minimum parametre sayısı
  - Etkili global arama
  - Hızlı yakınsama

## 💻 Kullanım

### Gereksinimler
```bash
pip install numpy matplotlib pandas
