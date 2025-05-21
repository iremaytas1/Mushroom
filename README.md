# Mushroom

# Mantarların Sınıflandırılması: Karar Ağaçları ile Zehirli ve Yenilebilir Mantarların Analizi

Bu proje, karar ağaçları kullanarak mantarların zehirli veya yenilebilir olarak sınıflandırılmasını amaçlar. 
UCI Machine Learning Repository'den alınan Mushroom veri seti üzerinde gerçekleştirilmiştir.<br>

# Proje Hakkında

Mantarların zehirli olup olmadığını belirlemek hayati önem taşır. Bu projede, mantarların çeşitli fiziksel özelliklerine dayanarak zehirli veya yenilebilir olduğunu tahmin eden bir makine öğrenmesi modeli geliştirilmiştir. Karar ağaçları ile oluşturulan model, %100 doğruluk oranıyla mükemmel bir sınıflandırma performansı göstermiştir.<br>

# Veri Seti

UCI Mushroom Veri Seti buradan alınmıştır : https://archive.ics.uci.edu/dataset/73/mushroom<br>
Veri Seti :  [mantar.zip](https://github.com/user-attachments/files/20369328/mantar.zip)<br>
8,124 örnek<br>
22 kategorik özellik<br>
İki sınıf: Zehirli (poisonous) ve Yenilebilir (edible)<br>


# Veri seti özellikleri şunları içerir:

Şapka şekli, yüzeyi ve rengi<br>
Koku<br>
Lamel (gill) özellikleri <br>
Sap yapısı ve özellikleri <br>
ve diğer morfolojik özellikler <br>

# Kurulum

Projeyi lokal ortamınızda çalıştırmak için aşağıdaki adımları izleyin:<br>
bash# Repository'yi klonlayın
git clone https://github.com/kullanıcıadınız/mantar-siniflandirma.git
cd mantar-siniflandirma

 Gerekli kütüphaneleri yükleyin

pip install -r requirements.txt

 Kodu çalıştırın

python mantar.py

Gereksinimler
Proje için gereken kütüphaneler:

pandas
numpy
matplotlib
seaborn
scikit-learn

# Kullanılan Yöntemler

Veri Ön İşleme:

Eksik değerlerin doldurulması
Kategorik değişkenlerin kodlanması (One-Hot Encoding)
Veri setinin eğitim ve test olarak bölünmesi (%70-%30)


Model Geliştirme:

Karar Ağacı Sınıflandırıcısı (DecisionTreeClassifier)
Hiperparametre optimizasyonu (max_depth, min_samples_leaf, min_samples_split)


Model Değerlendirme:

Doğruluk (Accuracy), Kesinlik (Precision), Duyarlılık (Recall), F1-skoru
Confusion Matrix analizi
Özellik önemliliği analizi



# Sonuçlar

Doğruluk (Accuracy): 1.0000 (%100)
Kesinlik (Precision): 1.0000
Duyarlılık (Recall): 1.0000
F1-skoru: 1.0000

Elde edilen mükemmel sonuçlar, mantarların fiziksel özelliklerinin, zehirli olup olmadıklarının belirlenmesinde güçlü bir gösterge olduğunu ortaya koymaktadır. 
Özellikle mantarın kokusu (odor), spor rengi (spore-print-color) ve lamel büyüklüğü (gill-size) en belirleyici faktörler olarak öne çıkmıştır.

# Görselleştirmeler

Proje çalıştırıldığında aşağıdaki görselleştirmeler otomatik olarak oluşturulur:

Karar Ağacı Görselleştirmesi (karar_agaci.png):
![karar_agaci](https://github.com/user-attachments/assets/b680cf35-9530-4a75-a816-3b774896c7bc)


Oluşturulan karar ağacı modelinin görsel temsili
Modelin karar verme sürecini ve önemli özellikleri gösterir


Confusion Matrix (confusion_matrix.png):
![confusion_matrix](https://github.com/user-attachments/assets/b9fd484d-d00b-44af-84d9-fc4eba1d03da)


Özellik Önemliliği (ozellik_onemliligi.png):
![ozellik_onemliligi](https://github.com/user-attachments/assets/7cab4e98-827d-4d87-840c-f25b410e6489)


En etkili 10 özelliği ve bunların göreceli önemini gösteren çubuk grafik



# Katkıda Bulunma
Katkılarınızı memnuniyetle karşılıyoruz! Lütfen bir pull request oluşturun veya iyileştirmeler için issues kısmından önerilerinizi paylaşın.

# İletişim
Sorularınız veya geri bildirimleriniz için:
E-posta: iremaytas47@gmail.com
GitHub: iremaytas1

