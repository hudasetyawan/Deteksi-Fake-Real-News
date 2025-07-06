# Deteksi-Fake-Real-News
UAS Kecerdasan Komputasional (TI22A)
``` phyton
# !pip install tensorflow scikit-learn pandas

# Import library utama
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
```
``` phyton
# Data berita dan label hoaks/asli
data = {
    "text": [
        "Pemerintah memberikan vaksin gratis",
        "Waspada! Ini obat palsu yang mematikan",
        "Presiden resmikan jembatan baru",
        "Virus ini dibuat oleh alien",
        "Kementerian bagikan bantuan sosial",
        "Ilmuwan temukan energi tak terbatas",
        "Vaksin menyebabkan kerusakan otak",
        "Presiden menghadiri upacara kenegaraan",
        "Dokter palsu sebarkan obat berbahaya",
        "Kampanye donor darah digelar di kota besar",
        "Manusia bisa hidup 200 tahun menurut penelitian",
        "Planet X akan menabrak bumi bulan depan",
        "Kementerian buka beasiswa baru untuk pelajar",
        "Makanan ini dapat menyembuhkan semua penyakit",
        "Banjir besar akibat eksperimen cuaca rahasia",
        "Guru besar UI memenangkan penghargaan internasional",
        "Alien membangun piramida Mesir",
        "Teknologi baru kurangi polusi udara",
        "Kiamat akan terjadi minggu depan menurut ramalan",
        "Pemerintah tingkatkan anggaran pendidikan"
    ],
    "label": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
              0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}

# Ubah ke DataFrame
df = pd.DataFrame(data)
```
``` phyton
# Parameter tokenizer
max_words = 1000  # Maksimal kata unik
max_len = 20      # Maksimal panjang kalimat (padding)

# Tokenisasi
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])  # Mengubah teks jadi angka
X = pad_sequences(sequences, maxlen=max_len)          # Padding ke panjang 20
y = np.array(df['label'])                             # Label jadi array
```
```phyton
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
```phyton
# Membuat model CNN
model = Sequential()

# Embedding layer: ubah token ke vektor angka
model.add(Embedding(input_dim=max_words, output_dim=50, input_length=max_len))

# Convolutional layer
model.add(Conv1D(64, 3, activation='relu'))

# Max pooling layer untuk ambil fitur terbaik
model.add(GlobalMaxPooling1D())

# Fully connected layer
model.add(Dense(10, activation='relu'))

# Output layer, pakai sigmoid karena hanya 2 kelas (hoaks/asli)
model.add(Dense(1, activation='sigmoid'))
```
```phyton
model.compile(
    optimizer='adam',                 # Cara belajar model: pakai Adam
    loss='binary_crossentropy',       # Cara menghitung error: cocok untuk 2 kelas
    metrics=['accuracy']              # Ukuran performa: pakai akurasi
)
```
```phyton
# Latih model
history = model.fit(
    X_train, y_train,
    epochs=10,                     # Latihan 10 kali
    validation_data=(X_test, y_test)
)
```
```phyton
import matplotlib.pyplot as plt

# Plot Akurasi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Testing')
plt.title('Grafik Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Testing')
plt.title('Grafik Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

```
```phyton
# Cek hasil akurasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Akurasi: {accuracy:.2f}')
```
