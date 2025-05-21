import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# Sessiz mod - sadece önemli çıktılar gösterilir
verbose = False

# Veri setini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]

try:
    mushroom_df = pd.read_csv(url, names=column_names)
except:
    try:
        mushroom_df = pd.read_csv('agaricus-lepiota.data', names=column_names)
    except Exception as e:
        print(f"Veri seti yüklenirken hata: {e}")
        exit(1)

if verbose:
    print(f"Veri seti boyutu: {mushroom_df.shape}")
    print(mushroom_df.head())

# Eksik değerleri kontrol et ve doldur
mushroom_df = mushroom_df.replace('?', np.nan)
missing_columns = mushroom_df.columns[mushroom_df.isna().any()].tolist()

for col in missing_columns:
    most_frequent = mushroom_df[col].mode()[0]
    mushroom_df[col].fillna(most_frequent, inplace=True)
    if verbose:
        print(f"'{col}' sütunundaki eksik değerler '{most_frequent}' ile dolduruldu")

# Sınıf isimlerini daha anlaşılır hale getir
mushroom_df['class_name'] = mushroom_df['class'].map({'e': 'edible', 'p': 'poisonous'})

# Sınıf dağılımını göster
print("\nSınıf Dağılımı:")
class_distribution = mushroom_df['class'].value_counts()
print(f"Yenilebilir (e): {class_distribution['e']}")
print(f"Zehirli (p): {class_distribution['p']}")

# Veri Hazırlama
# Hedef değişkeni ayır
y = (mushroom_df['class'] == 'e').astype(int)  # 1: yenilebilir, 0: zehirli

# Kategorik değişkenleri one-hot encoding ile kodla
X = pd.get_dummies(mushroom_df.drop(['class', 'class_name'], axis=1))

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Eğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti: {X_test.shape[0]} örnek")

# Karar Ağacı Modeli
print("\n--- Karar Ağacı Modeli Eğitiliyor ---")
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=10
)

# Modeli eğit
dt_model.fit(X_train, y_train)

# Tahmin yap
y_pred = dt_model.predict(X_test)

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performansı:")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"Kesinlik (Precision): {precision:.4f}")
print(f"Duyarlılık (Recall): {recall:.4f}")
print(f"F1-skoru: {f1:.4f}")

# Karar Ağacı Görselleştirme - sadece matplotlib kullanarak
plt.figure(figsize=(20, 10))
feature_names = X.columns.tolist()
class_names = ['Zehirli', 'Yenilebilir']

# Görselleştirme için daha basit bir ağaç oluştur
viz_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
viz_tree.fit(X_train, y_train)

# Ağacı çiz ve kaydet
plt.figure(figsize=(20, 10))
tree.plot_tree(viz_tree,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
plt.savefig('karar_agaci.png', dpi=300, bbox_inches='tight')
print("\nKarar ağacı görselleştirmesi 'karar_agaci.png' olarak kaydedildi.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# Özellik Önemliliği
feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': dt_model.feature_importances_
}).sort_values('Önem', ascending=False)

print("\nEn Önemli 5 Özellik:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"{row['Özellik']}: {row['Önem']:.4f}")

plt.figure(figsize=(12, 8))
sns.barplot(x='Önem', y='Özellik', data=feature_importance.head(10))
plt.title('Özellik Önemliliği')
plt.tight_layout()
plt.savefig('ozellik_onemliligi.png', dpi=300, bbox_inches='tight')

print("\nTüm görseller ve analizler başarıyla tamamlandı.")