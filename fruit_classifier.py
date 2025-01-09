import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import cv2
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

def load_images_without_resize(directory):
    """
    Resimleri belirtilen klasörden yükler, boyutlarını korur ve etiketleri döndürür.
    
    Args:
        directory (str): Resimlerin olduğu klasör.
    
    Returns:
        Tuple[list, list]: Resimler (liste) ve etiketler (liste).
    """
    images = []
    labels = []
    for fruit_dir in os.listdir(directory):
        fruit_path = os.path.join(directory, fruit_dir)
        if os.path.isdir(fruit_path):
            for img_file in os.listdir(fruit_path):
                img_path = os.path.join(fruit_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)  # Resmi olduğu gibi ekle (boyut korundu)
                    labels.append(fruit_dir)
    return images, labels

def resize_images(images, size):
    return np.array([cv2.resize(img, size) for img in images])

# Veri yolları
data_path = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
train_path = os.path.join(data_path, "Training")
validation_path = os.path.join(data_path, "Validation")
test_path = os.path.join(data_path, "Test")

# Resimleri yükleme
X_train_images_original, y_train_labels = load_images_without_resize(train_path)
X_validation_images_original, y_validation_labels = load_images_without_resize(validation_path)
X_test_images_original, y_test_labels = load_images_without_resize(test_path)

# NumPy dizisi oluşturma (boyutlar farklı olduğu için dtype=object)
X_train_images_original = np.array(X_train_images_original, dtype=object)
X_validation_images_original = np.array(X_validation_images_original, dtype=object)
X_test_images_original = np.array(X_test_images_original, dtype=object)

print(f"Train set shape: {X_train_images_original.shape}")
print(f"Validation set shape: {X_validation_images_original.shape}")
print(f"Test set shape: {X_test_images_original.shape}")

# Parametre kombinasyonları
photo_sizes = [(64, 64), (32, 32)]
pca_components = [5, 25, 50]
dropout_rates = [None, 0.3, 0.6]
ml_param_combinations = list(product(photo_sizes, pca_components))
nn_param_combinations = list(product(photo_sizes, dropout_rates))

# Algoritmalar
algorithms = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
}

# Sonuçları kaydetmek için DataFrame
results = []

# Etiketleri kodla
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_labels)
y_validation = label_encoder.transform(y_validation_labels)
y_test = label_encoder.transform(y_test_labels)

# ML algoritmaları için eğitim ve test döngüsü
for photo_size, n_pca in ml_param_combinations:
    print(f"Running ML models with Photo Size: {photo_size}, PCA: {n_pca}")
    
    # Verileri yeniden boyutlandır
    X_train_images = resize_images(X_train_images_original, photo_size)
    X_validation_images = resize_images(X_validation_images_original, photo_size)
    X_test_images = resize_images(X_test_images_original, photo_size)
    
    # Verileri düzleştir ve ölçeklendir
    X_train = X_train_images.reshape(X_train_images.shape[0], -1)
    X_validation = X_validation_images.reshape(X_validation_images.shape[0], -1)
    X_test = X_test_images.reshape(X_test_images.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # PCA uygula
    pca = PCA(n_components=n_pca)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_validation_pca = pca.transform(X_validation_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    for name, model in algorithms.items():
        start_time = time.time()
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy')
        validation_accuracy = model.score(X_validation_pca, y_validation)
        
        # Performans metrikleri
        test_accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        duration = time.time() - start_time

        # Sonuçları kaydet
        results.append({
            "Algorithm": name,
            "Photo Size": photo_size,
            "PCA Components": n_pca,
            "Dropout": None,
            "Validation Accuracy": validation_accuracy,
            "Test Accuracy": test_accuracy,
            "CV Accuracy": np.mean(cv_scores),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Time (s)": duration
        })


# ANN ve CNN için eğitim ve test döngüsü
for photo_size, dropout in nn_param_combinations:
    print(f"Running NN models with Photo Size: {photo_size}, Dropout: {dropout}")

    # Verileri yeniden boyutlandır
    X_train_images = resize_images(X_train_images_original, photo_size)
    X_validation_images = resize_images(X_validation_images_original, photo_size)
    X_test_images = resize_images(X_test_images_original, photo_size)

    # Normalizasyon
    X_train = X_train_images / 255.0
    X_validation = X_validation_images / 255.0
    X_test = X_test_images / 255.0
    y_train_categorical = to_categorical(y_train)
    y_validation_categorical = to_categorical(y_validation)
    y_test_categorical = to_categorical(y_test)

    # ANN Modeli
    start_time = time.time()
    ann = Sequential([
        Dense(128, activation='relu', input_dim=np.prod(photo_size) * 3),
        Dropout(dropout if dropout else 0.0),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_validation_flat = X_validation.reshape(X_validation.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    ann.fit(X_train_flat, y_train_categorical, validation_data=(X_validation_flat, y_validation_categorical), epochs=5, batch_size=32, verbose=0)
    ann_validation_accuracy = ann.evaluate(X_validation_flat, y_validation_categorical, verbose=0)[1]
    ann_test_accuracy = ann.evaluate(X_test_flat, y_test_categorical, verbose=0)[1]
    
    # Test set tahminleri ve metrikler
    y_pred_ann = np.argmax(ann.predict(X_test_flat, verbose=0), axis=1)
    precision = precision_score(y_test, y_pred_ann, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_ann, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_ann, average='weighted', zero_division=0)
    duration = time.time() - start_time

    # Sonuçları kaydet
    results.append({
        "Algorithm": "ANN",
        "Photo Size": photo_size,
        "PCA Components": None,
        "Dropout": dropout,
        "Validation Accuracy": ann_validation_accuracy,
        "Test Accuracy": ann_test_accuracy,
        "CV Accuracy": None,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Time (s)": duration
    })

    # CNN Modeli
    start_time = time.time()
    cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(photo_size[0], photo_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout if dropout else 0.0),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train, y_train_categorical, validation_data=(X_validation, y_validation_categorical), epochs=5, batch_size=32, verbose=0)
    cnn_validation_accuracy = cnn.evaluate(X_validation, y_validation_categorical, verbose=0)[1]
    cnn_test_accuracy = cnn.evaluate(X_test, y_test_categorical, verbose=0)[1]

    # Test set tahminleri ve metrikler
    y_pred_cnn = np.argmax(cnn.predict(X_test, verbose=0), axis=1)
    precision = precision_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
    duration = time.time() - start_time

    # Sonuçları kaydet
    results.append({
        "Algorithm": "CNN",
        "Photo Size": photo_size,
        "PCA Components": None,
        "Dropout": dropout,
        "Validation Accuracy": cnn_validation_accuracy,
        "Test Accuracy": cnn_test_accuracy,
        "CV Accuracy": None,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Time (s)": duration
    })


# Sonuçları DataFrame'e çevir
results_df = pd.DataFrame(results)

# Ondalıklı sütunları formatla
float_columns = ["Validation Accuracy", "Test Accuracy", "CV Accuracy", "Precision", "Recall", "F1 Score", "Time (s)"]
results_df[float_columns] = results_df[float_columns].applymap(lambda x: round(x, 4) if pd.notnull(x) else x)

# Sonuçları kaydet
results_df.to_csv("model_results_with_validation.csv", index=False)

results_df["Photo Size"] = results_df["Photo Size"].apply(lambda x: f"{x[0]}x{x[1]}")

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Dropout", y="Test Accuracy", hue="Algorithm")
plt.title("Test Accuracy by Dropout Rate and Algorithm")
plt.ylabel("Test Accuracy")
plt.xlabel("Dropout Rate")
plt.legend(title="Algorithm")
plt.show()

# ANN ve CNN sonuçlarını filtrele
nn_results = results_df[(results_df["Algorithm"].isin(["ANN", "CNN"]))]

# Dropout değeri 'None' olanları düzelt
nn_results["Dropout"] = nn_results["Dropout"].fillna("None")

# Barplot oluştur
plt.figure(figsize=(10, 6))
sns.barplot(
    data=nn_results,
    x="Dropout",
    y="Test Accuracy",
    hue="Algorithm"
)

# Eksenler ve başlık
plt.title("Effect of Dropout on Test Accuracy (ANN & CNN)", fontsize=14)
plt.xlabel("Dropout Rate", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.legend(title="Algorithm", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

#%%

# Random Forest ile çalıştırma
photo_size = (64, 64)
pca_components = 50

# Resimleri yeniden boyutlandırma
X_train_images_resized = resize_images(X_train_images_original, photo_size)
X_validation_images_resized = resize_images(X_validation_images_original, photo_size)
X_test_images_resized = resize_images(X_test_images_original, photo_size)

# Verileri düzleştir ve ölçeklendir
X_train = X_train_images_resized.reshape(X_train_images_resized.shape[0], -1)
X_validation = X_validation_images_resized.reshape(X_validation_images_resized.shape[0], -1)
X_test = X_test_images_resized.reshape(X_test_images_resized.shape[0], -1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# PCA uygula
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_validation_pca = pca.transform(X_validation_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Random Forest modelini eğitme
rf_model = RandomForestClassifier()
rf_model.fit(X_train_pca, y_train)

# Test verisiyle tahmin yapma
y_pred_rf = rf_model.predict(X_test_pca)

# Confusion Matrix hesaplama ve görselleştirme
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#%%

# ANN modeli
photo_size = (64, 64)
dropout = None

# Verileri yeniden boyutlandırma ve normalizasyon
X_train_images_resized = resize_images(X_train_images_original, photo_size)
X_validation_images_resized = resize_images(X_validation_images_original, photo_size)
X_test_images_resized = resize_images(X_test_images_original, photo_size)

X_train = X_train_images_resized / 255.0
X_validation = X_validation_images_resized / 255.0
X_test = X_test_images_resized / 255.0
y_train_categorical = to_categorical(y_train)
y_validation_categorical = to_categorical(y_validation)
y_test_categorical = to_categorical(y_test)

# ANN Modeli
ann = Sequential([
    Dense(128, activation='relu', input_dim=np.prod(photo_size) * 3),
    Dropout(dropout if dropout else 0.0),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = ann.fit(X_train.reshape(X_train.shape[0], -1), y_train_categorical, validation_data=(X_validation.reshape(X_validation.shape[0], -1), y_validation_categorical), epochs=5, batch_size=32, verbose=0)

# Training/Validation Accuracy ve Loss grafiklerini çizme
# Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for ANN', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for ANN', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

#%%

# CNN modeli
photo_size = (64, 64)
dropout = None

# Verileri yeniden boyutlandırma ve normalizasyon
X_train_images_resized = resize_images(X_train_images_original, photo_size)
X_validation_images_resized = resize_images(X_validation_images_original, photo_size)
X_test_images_resized = resize_images(X_test_images_original, photo_size)

X_train = X_train_images_resized / 255.0
X_validation = X_validation_images_resized / 255.0
X_test = X_test_images_resized / 255.0
y_train_categorical = to_categorical(y_train)
y_validation_categorical = to_categorical(y_validation)
y_test_categorical = to_categorical(y_test)

# CNN Modeli
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(photo_size[0], photo_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(dropout if dropout else 0.0),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history_cnn = cnn.fit(X_train, y_train_categorical, validation_data=(X_validation, y_validation_categorical), epochs=5, batch_size=32, verbose=0)

# Training/Validation Accuracy ve Loss grafiklerini çizme
# Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for CNN', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.figure(figsize=(12, 6))
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for CNN', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


