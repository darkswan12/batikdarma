import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Memuat model yang sudah dilatih
model = load_model('D:/dokumen/PI/batikdarma/BatikDetection.h5')

# Mendefinisikan label berdasarkan kelas yang digunakan saat pelatihan
labels = ["Batik Betawi", "Batik Kawung", "Batik Megamendung", "Batik Parang", "Batik Sekar Jagad"]

# Menampilkan judul aplikasi Streamlit
st.title("Aplikasi untuk Klasifikasi Batik")

# Menambahkan deskripsi di bawah judul
st.write("Aplikasi ini dapat mendeteksi jenis batik berikut: Betawi, Kawung, Megamendung, Parang, dan Sekar Jagad."
         "Silakan unggah gambar batik untuk mengetahui jenisnya.")


# Mengunggah gambar
uploaded_file = st.file_uploader("Unggah Gambar Batik (format 'jpg','jpeg' dan 'png') ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption="Hasil Unggahan", use_container_width=True)

    # Mengonversi gambar menjadi RGB jika belum RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Memproses gambar agar sesuai dengan ukuran input model
    img = img.resize((224, 224))  # ukuran dalam preprocessing gambar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi jika diperlukan
    
    # Melakukan prediksi menggunakan model
    prediction = model.predict(img_array)
    pred_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Menampilkan hasil prediksi
    st.write(f"Prediksi: **{pred_label}**")
    st.write(f"Probabilitas Keyakinan: **{confidence:.2f}%**")