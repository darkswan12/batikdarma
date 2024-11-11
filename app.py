import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Memuat model yang sudah dilatih
model = load_model('D:/dokumen/PI/batikdarma/BatikDetection.h5')

# Mendefinisikan label berdasarkan kelas yang digunakan saat pelatihan
labels = ["Batik Betawi", "Batik Kawung", "Batik Megamendung", "Batik Parang", "Batik Sekar Jagad"]

# Mendefinisikan deskripsi untuk setiap jenis batik
batik_descriptions = {
    "Batik Betawi": "Batik Betawi mencerminkan kekayaan budaya masyarakat Betawi yang beragam, menggunakan warna-warna cerah dan kontras. Motif-motifnya sering kali mengangkat tema alam dan tradisi lokal, seperti bunga, burung, dan ondel-ondel, yang merupakan ikon budaya Betawi. Selain itu, pola dalam batik Betawi biasanya memiliki elemen simetris dan terstruktur, menggambarkan keseimbangan dalam keberagaman budaya Betawi yang khas.",
    "Batik Kawung": "Batik Kawung memiliki pola geometris yang sederhana namun penuh makna, dengan motif lingkaran-lingkaran kecil yang menyerupai buah kolang-kaling. Susunan pola yang berulang ini melambangkan harmoni, kesucian, dan kebijaksanaan. Secara tradisional, Batik Kawung sering dipakai oleh kalangan bangsawan Jawa sebagai simbol kekuatan dan keanggunan, mencerminkan sifat bijaksana dan integritas moral.",
    "Batik Megamendung": "Batik Parang adalah salah satu motif batik tertua di Indonesia yang terkenal dengan pola garis diagonal berulang menyerupai ombak laut atau pedang (parang). Motif ini melambangkan kekuatan, keberanian, dan semangat pantang menyerah. Pada masa lalu, Batik Parang sering digunakan oleh keluarga kerajaan sebagai simbol keberanian dan keteguhan dalam menjalankan tanggung jawab.",
    "Batik Parang": "Batik Parang adalah motif batik tertua yang bercirikan pola diagonal menyerupai pedang atau ombak, melambangkan keberanian dan tekad pantang menyerah.",
    "Batik Sekar Jagad": "Batik Sekar Jagad dikenal dengan pola yang rumit dan indah, menyerupai peta atau pulau-pulau yang terhubung. Nama “Sekar Jagad” berasal dari kata “sekar” (bunga) dan “jagad” (dunia), melambangkan keindahan dan keragaman dunia. Pola ini menggambarkan kesatuan dalam keragaman budaya Nusantara, dengan warna-warna yang harmonis dan komposisi yang detail, menjadikannya simbol keindahan dan persatuan."
}

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

    # Menampilkan deskripsi batik yang diklasifikasikan
    st.write("### **Deskripsi Batik:**")
    st.write(batik_descriptions.get(pred_label, "**Deskripsi tidak tersedia.**"))