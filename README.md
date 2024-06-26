<h1 align="center"> KuyFutsal's AI Features and Documentations Until Deployments & Integrations <br /> <i>(NaiveBayes Machine Learning Model for Chatbot & 'face-recognition' Library with FaceNet, Dlib, etc for FaceRec) </i></h1>
<p> 
Proyek ini menggunakan IBM Watson Discovery dan Watsonx Assistant untuk membangun chatbot cerdas. Data yang sudah dipreproses dilatih menggunakan Watson Discovery untuk klasifikasi teks, yang membantu dalam mengidentifikasi topik, sentimen, dan maksud pengguna. Model yang sudah dilatih kemudian diintegrasikan ke dalam Watsonx Assistant, memungkinkan chatbot untuk memproses input pengguna, memahami maksudnya, dan memberikan respons yang relevan. Deployment da Integrasi web menggunakan WatsonXAssistant  dari IBM Cloud.
</p>
<h2 align="center"> Python Flask API to Code Engine </h2>
<p align="center"> Deploy kontainer Docker yang berisi Flask API untuk load model Machine Learning berbasis pkl ke layanan IBM Cloud Code Engine</p>

<div align="center">
    <!-- Your badges here -->
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/IBM%20Cloud-1261FE?style=for-the-badge&logo=IBM%20Cloud&logoColor=white">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white">
</div>

## A. Teams
- Edwin Prayoga (SCRUM Master, Design Researcher, ML Engineer)
- Marco Philips Sirait (Data Engineer, ML Engineer, ML Ops)
- Bagas Hilmi Arib (Data Engineer)
- Noneng Ismaryanti (Machine Learning Ops)

## B. Syarat-syarat
- Perangkat dengan sistem Windows 10 64-bit atau lebih baru (Agar Maksimal)
- RAM minimal 4GB
- Penyimpanan tersisa minimal 5GB-20GB
- Docker Desktop (download [disini](https://www.docker.com/products/docker-desktop/)) +- 500MB
- Python 3.8+ 
- Akun Docker Hub (daftar [disini](https://hub.docker.com/signup))
- Akun IBM Cloud

## C. Idea Background

### 1. Theme
<b>Service/Tourism & A bit of Health</b>
<br/><br/>
KuyFutsal adalah proyek yang bertujuan untuk mempermudah akses dan informasi penyewaan lapangan futsal melalui sebuah platform berbasis website. Tema utama dari proyek ini adalah "Digitalisasi Layanan Penyewaan Lapangan Futsal dengan Teknologi AI". Dengan mengintegrasikan teknologi kecerdasan buatan (AI) seperti chatbot dan pengenalan wajah, KuyFutsal bertujuan untuk memberikan layanan yang lebih aman, cepat, dan interaktif bagi pengguna.

### 2. Problem
Awal mula kami dalam menentukan projek kami yaitu adalah sebuah produk digital sebagai solusi dalam permasalahan sehari-hari. Dari tujuan awal tersebut di temukanlah sebuah ide media penyewaan lapangan futsal berbasis website yang bernama “KuyFutsal”. Alasan kami dalam memilih ide tersebut karena kurangnya informasi penyedia lapangan futsal yang ada. Selain itu, kompetitor sejenis penyewaan lapangan olahraga juga tergolong sedikit sehingga membuat kami lebih yakin dalam membuat website penyewaan lapangan yang berfokus pada olahraga futsal.
<br/><br/>
![Theme](Chatbot\Documentation\image_008.png)
<br/><br/>
Belum lagi Permasalahan mengenai banyaknya orang-orang yang masih kurang peduli akan kesehatannya, membuat kami ingin membantu mereka merekonsiderasikan pilihan mereka untuk melakukan yang terbaik buat tubuh mereka.
<br/><br/>
![Theme](Rndm_Readme_Imgs\Screenshot%202024-06-25%20233015.png)

### 3. Solution
Untuk mencapai tujuan agar website KuyFutsal dapat berguna dan menarik untuk digunakan masyarakat luas, kami menyematkan fitur unggulan pada website kami sendiri demi meningkatkan nilai jual produk kami, yaitu dengan menggunakan teknologi AI face recognition atau pengenalan wajah dan AI chatbot. Fitur AI Face Recognition digunakan sebagai sistem keamanan dalam melakukan verifikasi keamanan dalam pemesanan. Fitur chatbot digunakan sebagai customer service website yang dapat menyediakan informasi seputar futsal maupun tata cara penggunaan website.
<br/><br/>
Hasil yang diharap dari project website KuyFutsal ini yaitu mampu menjadi sebuah produk digital yang menarik dengan adanya fitur AI Face Recognition dan Chatbot yang menjadi nilai tambahan. Sehingga nantinya website kami tidak hanya sebagai media pemesanan maupun informasi, namun juga dapat memberikan layanan yang interaktif dan menarik.

## D. Dataset and Algorithm

### 1. Dataset
- <b>Data Collection</b> <br />
Dataset yang kami gunakan adalah dataset berupa file ‘.csv/.json’ yang berisikan ‘intents’, dan didalam intents terdapat tags yang mengandung question/pattern (variasi user input) dan answer/response yang berbeda-beda. Data tersebut kami kumpulkan melalui research mendalam mengenai topik-topik seperti Futsal dan seputarnya, serta mengenai website kami dan cara melayani kustomer. Setelah research kami kumpulkan pertanyaanpertanyaan yang sekiranya akan ditanyakan oleh user berdasarkan topik-topik tadi.
<br/><br/>
Pengumpulannya menggunakan sistem Automated yang saya buat dan terlampir di sebuah Google Colab notebook. Dimana kami juga menggunakan sebuah model Parafrase dan model Translasi untuk membantu dalam penggandaan, penduplikasian, dan pengumpulan data guna untuk memvariasikan pattern dan response data. Lalu kami juga menggunakan sebuah modul buatan sendiri dengan folder bernama ‘util’ yang berisikan file parser.py yang berguna untuk mamparser file ‘.json’ nya nanti untuk di training.
<br/><br/>
FaceRec tidak memerlukan pengumpulan data yang spesifik sama sekali, karena tinggal menerima input user.

<hr>

Model Parafrase Pegasus from HF:
```python
# Nama model yang akan digunakan
model_name = 'tuner007/pegasus_paraphrase'

# Tentukan perangkat keras yang akan digunakan (GPU jika tersedia, jika tidak CPU)
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inisialisasi tokenizer dari model Pegasus yang telah diunduh sebelumnya
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Inisialisasi model Pegasus untuk generasi kondisional dari model yang telah diunduh sebelumnya
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
```
Model Translasi NLP Opus from HF:
```python
# Inisialisasi pipeline untuk terjemahan dari bahasa Inggris ke bahasa Indonesia
transpipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
```
Main Code:
```python
def main():
    intents = []
    stop_words = {"Quit", "No", "Escape", "Q", "Esc", "Stop", "Done", "quit", "no", "escape", "q", "esc", "stop", "done"}

    while True:
        # Meminta input tag dari pengguna
        tag = input("Masukkan tag: ").strip()
        # Keluar dari loop jika input adalah salah satu stop word
        if tag.lower() in stop_words:
            break

##################   PATTERNS   ##################

        patterns = []
        while True:
            # Meminta input pola dari pengguna
            pattern = input("Masukkan pola (atau ketik 'done' untuk menghentikan penambahan pola): ").strip()
            # Keluar dari loop jika input adalah salah satu stop word
            if pattern.lower() in stop_words:
                break
            # Terjemahkan pola ke bahasa Indonesia
            indopat = transpipe(pattern)
            indo_pattern = indopat[0]['translation_text']
            patterns.append(indo_pattern)

            # Dapatkan parafrase dari pola
            pattern_paraphrases = get_paraphrase(pattern)
            patterns.append(pattern)
            patterns.extend(pattern_paraphrases)

            # Terjemahkan parafrase pola ke bahasa Indonesia
            pattern_translation = []
            for translate in pattern_paraphrases:
                result = transpipe(translate)
                translation_pattern = result[0]['translation_text']
                pattern_translation.append(translation_pattern)
            patterns.extend(pattern_translation)

            # Hapus duplikat dari daftar pola
            patterns = remove_duplicates(patterns)

##################   RESPONSES   ##################

        responses = []
        while True:
            # Meminta input tanggapan dari pengguna
            response = input("Masukkan tanggapan (atau ketik 'done' untuk menghentikan penambahan tanggapan): ").strip()
            # Keluar dari loop jika input adalah salah satu stop word
            if response.lower() in stop_words:
                break
            # Terjemahkan tanggapan ke bahasa Indonesia
            indores = transpipe(response)
            indo_response = indores[0]['translation_text']
            responses.append(response)

            # Dapatkan parafrase dari tanggapan
            response_paraphrases = get_paraphrase(response)
            responses.append(indo_response)
            responses.extend(response_paraphrases)

            # Terjemahkan parafrase tanggapan ke bahasa Indonesia
            response_translation = []
            for translate in response_paraphrases:
                result = transpipe(translate)
                translation_response = result[0]['translation_text']
                response_translation.append(translation_response)
            responses.extend(response_translation)

            # Hapus duplikat dari daftar tanggapan
            responses = remove_duplicates(responses)

##################   DATASET FILE   ##################

        # Buat objek intent dengan tag, pola, dan tanggapan
        intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }

        # Tambahkan intent ke daftar intents
        intents.append(intent)
        print(f"Ditambahkan intent: {intent}")
        print(intents)

        # Simpan intents ke dalam file intents.json
        with open('intents.json', 'w') as f:
            json.dump({"intents": intents}, f, indent=4)
            print("Intents disimpan ke intents.json")
```

<hr>

- <b>Data Cleaning</b> <br />
Setelah kami melakukan pengumpulan data, langkah pertama yang kami lakukan adalah membersihkan data secara manual. Proses ini melibatkan penghapusan baris kalimat pattern atau response yang tidak masuk akal. Selain itu, kami juga memperpadat pattern dan response untuk masing-masing tag dalam intents agar data menjadi lebih terstruktur dan mudah diolah. 
<br/><br/>
Dalam tahap ini, kami menggunakan library pandas untuk membantu membersihkan dan memeriksa data. Pandas memungkinkan kami untuk dengan mudah mendeteksi dan menghapus baris yang memiliki missing values atau nilai yang hilang, sehingga data yang kami gunakan untuk analisis selanjutnya menjadi lebih bersih dan akurat. Melalui proses pembersihan ini, kami memastikan bahwa data yang kami miliki berkualitas tinggi dan siap digunakan untuk tahap analisis lebih lanjut.
<br/><br/>
FaceRec tidak memerlukan cleaning data yang spesifik sama sekali, karena tinggal menerima input user.

<hr>

Cleaning Data Secara Manual Docs:
<br/><br/>
![Dataset](Chatbot\Documentation\image_001.png)

<hr>

- Data Pre-Processing<br />
Dalam proses preprocessing data, kami menerapkan beberapa langkah penting untuk mempersiapkan teks agar siap digunakan dalam analisis lebih lanjut. Pertama, kami mengkonversi seluruh teks ke dalam huruf kecil untuk memastikan konsistensi dan menghindari perbedaan yang disebabkan oleh kapitalisasi. Selanjutnya, kami menghapus tanda baca yang tidak diperlukan seperti titik, koma, dan tanda tanya, tetapi tetap mempertahankan tanda apostrof dan tanda hubung yang mungkin memiliki arti khusus dalam konteks kalimat. Kami juga membersihkan teks dari karakter yang tidak diinginkan dan menghilangkan spasi ekstra untuk memastikan teks yang lebih rapi dan terstruktur. Proses ini mencakup penggunaan fungsi-fungsi seperti case folding, penghapusan karakter yang tidak diinginkan, dan lemmatisasi untuk mengubah kata-kata menjadi bentuk dasarnya. Langkah terakhir dalam preprocessing adalah memastikan hanya huruf alfabet dan apostrof yang tersisa dalam teks, kemudian menggabungkan kembali kata-kata tersebut untuk menjaga keutuhan kalimat. Dengan menerapkan serangkaian langkah preprocessing ini, kami memastikan bahwa data yang dihasilkan siap untuk analisis selanjutnya dengan kualitas yang lebih baik.
<br/><br/>
<i>Kami juga melakukan Vektorisasi!</i>
<br/><br/>
FaceRec tidak memerlukan Preprocess data yang spesifik sama sekali, karena tinggal menerima input user. Lebih tepatnya FaceRec didalam library python 'face-recognition' sudah memiliki sistem preprocessing gambarnya sendiri dimana dia memiliki fungsi encode gambar untuk proses gambar menjadi ukuran yang diterima dan library face_recognition supports BGR format dari images dan masih banyak lagi.

<hr>

Semua Preprocessing Data Chatbot Kami, Lebih Tepatnya Text-Preprocessing:
```python
# Main preprocessing
def preprocess_text(text):
    """
    Fungsi yang digunakan untuk melakukan praproses
    """
    # konversi ke lowercase
    text = text.lower()
    # menghapus tanda baca
    unwantedchars = ["'", "-"]
    tandabaca = tuple(c for c in string.punctuation if c not in unwantedchars)
    text = ''.join(ch for ch in text if ch not in tandabaca)
    return text

def last_preprocess(text):
    text = case_folding(text)
    text = remove_unwanted_chars(text)
    # text = remove_numbers(text)
    text = remove_extra_whitespace(text)
    text = lemmatization(text)
    return text

# Preprocessing function YANG AKAN DIGUNAKAN
def end_preprocess(text):
    text = re.sub('[^a-zA-Z\'-]', ' ', text)  # Keep only alphabets and apostrophes
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Split into words
    text = " ".join(text)  # Rejoin words to ensure clean spacing
    return text
```

Berikut Visualisasi Data AKHIR (Yang Pertama Udah Ada di Atas Tadi):<br />
![Datasets](Chatbot\Documentation\image_035.png)
![Datasets](Chatbot\Documentation\image_036.png)

<hr>

### 2. Algorithm
Algoritma yang dipakai tentu bisa dilihat dari bentuk dataset yang sudah kami buat, kami nmenggunakan Algoritma Klasifikasi Intents, Algoritma Prediksi Klasifikasi, dan juga menggunakan/menerapkan Algoritma NLP (anggap salah satu algoritma) untuk melakukan Entity Recognizer. Ada juga Intent Classification sebagai tujuan/produk/fitur utama.
<br/><br/>
Dalam pengembangan model chatbot berbasis machine learning, kami memilih menggunakan algoritma Naive Bayes MultinomialNB dari library Scikit-learn. Algoritma ini dipilih karena memiliki kemampuan yang sangat baik dalam mengklasifikasikan data teks, yang merupakan jenis data utama dalam percakapan chatbot. Naive Bayes MultinomialNB dirancang khusus untuk menangani data yang berbentuk frekuensi atau jumlah kemunculan suatu kata. Hal ini sangat relevan untuk aplikasi chatbot di mana analisis frekuensi kata-kata dalam input pengguna diperlukan untuk menentukan respons yang tepat.

- <b>Framework AI</b> <br />
Kami memutuskan untuk menggunakan Scikit-Learn sebagai kerangka kerja kecerdasan buatan utama kami dalam membangun model jaringan saraf yang kompleks. Alasan utamanya ialah karena Machine Learning Engineer kami, Marco Philips Sirait sudah berpengalaman dengan Framework tersebut, alasan lainnya yaitu memilih framework Scikit-learn dalam pengembangan chatbot berbasis machine learning menawarkan berbagai keuntungan yang signifikan. Scikit-learn adalah library yang terkenal karena kesederhanaannya dan kemudahan penggunaannya, yang sangat berguna dalam proses pengembangan cepat. Framework ini menyediakan beragam algoritma machine learning yang sudah teroptimasi, mulai dari klasifikasi, regresi, clustering, hingga reduksi dimensi, memungkinkan pengembang untuk memilih model yang paling sesuai dengan kebutuhan chatbot mereka. Selain itu, Scikit-learn memiliki dokumentasi yang sangat baik dan komunitas yang aktif, sehingga memudahkan pengembang untuk menemukan solusi atas permasalahan teknis yang mereka hadapi. Dengan dukungan terhadap integrasi dengan library lain seperti NumPy, SciPy, dan Pandas, Scikit-learn memungkinkan penanganan dan manipulasi data yang lebih efisien. Kombinasi dari performa yang kuat, fleksibilitas, dan dukungan komunitas membuat Scikit-learn menjadi pilihan yang ideal untuk mengembangkan chatbot berbasis machine learning yang andal dan efisien.

<hr>

Imported Packages & Libraries:
```python
# import library untuk serialisasi objek Python
import pickle

# import CountVectorizer dari sklearn untuk mengubah koleksi dokumen teks menjadi vektor fitur
from sklearn.feature_extraction.text import CountVectorizer

# import MultinomialNB dari sklearn untuk menerapkan klasifikasi Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# import make_pipeline dari sklearn untuk membuat pipeline yang menggabungkan beberapa langkah pemrosesan data
from sklearn.pipeline import make_pipeline
```
Library Utils -> JsonParser Buatan Sendiri Untuk Parser Data:
```python
import json
import pandas as pd
from random import choice

# Kelas untuk mengurai data dari file JSON menjadi DataFrame dan menyediakan fungsi utilitas lainnya
class JSONParser:
    def __init__(self):
        self.text = []  # List untuk menyimpan teks dari pola
        self.intents = []  # List untuk menyimpan intent
        self.responses = {}  # Dictionary untuk menyimpan tanggapan berdasarkan intent

    # Metode untuk mengurai data dari file JSON menjadi DataFrame
    def parse(self, json_path):
        # Buka file JSON dan muat datanya
        with open(json_path) as data_file:
            self.data = json.load(data_file)

        # Iterasi melalui setiap intent dalam data JSON
        for intent in self.data['intents']:
            # Iterasi melalui setiap pola dalam intent dan tambahkan ke list teks dan intents
            for pattern in intent['patterns']:
                self.text.append(pattern)
                self.intents.append(intent['tag'])
            # Iterasi melalui setiap tanggapan dalam intent dan tambahkan ke dictionary tanggapan
            for resp in intent['responses']:
                if intent['tag'] in self.responses.keys():
                    self.responses[intent['tag']].append(resp)
                else:
                    self.responses[intent['tag']] = [resp]

        # Buat DataFrame dari teks pola dan intents
        self.df = pd.DataFrame({'text_input': self.text,
                                'intents': self.intents})

        # Cetak informasi tentang DataFrame yang dibuat
        print(
            f"[INFO] Data JSON dikonversi menjadi DataFrame dengan bentuk: {self.df.shape}")

    # Metode untuk mendapatkan DataFrame yang telah dibuat
    def get_dataframe(self):
        return self.df

    # Metode untuk mendapatkan tanggapan acak berdasarkan intent
    def get_response(self, intent):
        return choice(self.responses[intent])
```

<hr>

- <b>Model AI</b> <br />
Setelah nelihat code2 & gambar2 diatas, bisa dilihat kami memilih menggunakan algoritma Naive Bayes MultinomialNB dari library Scikit-learn dalam pembuatan model chatbot karena algoritma ini sangat cocok untuk melakukan klasifikasi data bertipe teks. Multinomial Naive Bayes dirancang khusus untuk menangani data yang berbentuk frekuensi atau jumlah kemunculan suatu kata, yang merupakan karakteristik umum dalam data teks seperti percakapan chatbot. Algoritma ini bekerja dengan sangat efisien dalam mengklasifikasikan teks ke dalam kategori yang telah ditentukan, sehingga memungkinkan chatbot untuk memberikan respons yang relevan dan akurat terhadap input dari pengguna. Selain itu, Naive Bayes Multinomial juga dikenal dengan kecepatan dan kemampuannya dalam mengatasi data yang sangat besar, yang sering ditemukan dalam aplikasi chatbot. 
<br/><br/>
Implementasi MultinomialNB di Scikit-learn memanfaatkan optimisasi yang sangat baik sehingga proses pelatihan dan prediksi dapat dilakukan dengan cepat. Kemudahan dalam penggunaan Scikit-learn, yang menawarkan antarmuka yang intuitif dan dokumentasi yang lengkap, juga menjadi faktor penting dalam mempercepat pengembangan model chatbot. Tidak hanya itu, komunitas aktif di sekitar Scikit-learn menyediakan berbagai sumber daya dan dukungan, membantu pengembang mengatasi berbagai tantangan teknis yang mungkin dihadapi selama pengembangan. Kombinasi dari algoritma yang tepat guna, performa yang tinggi, dan ekosistem pendukung yang kuat menjadikan Scikit-learn dan algoritma Multinomial Naive Bayes sebagai pilihan ideal untuk mengembangkan model chatbot yang efektif dan responsif

- <b>Model Building, Training, and Evaluation</b> <br />
Vektorisasi -> Model MultinomialNB -> Fitting & Training -> Testing -> Evaluation. Berikut kesimpulan Model dan data kami dalam bentuk diagram alir. kami menggunakan model MultinomialNB untuk melakukan klasifikasi intents pada Chatbot kami. Sebenarnya ini juga sudah termasuk ke bagian model evaluasi dikarenakan memiliki prediction probability, namun untuk lebih jelas dan lengkapnya akan dijelaskan di section berikutnya Model Evaluation. Model yang digunakan dalam skrip ini adalah sebuah pipeline yang menggunakan CountVectorizer untuk mengubah teks menjadi vektor fitur dan Multinomial Naive Bayes sebagai model klasifikasi untuk mengidentifikasi niat atau intent dari teks masukan pengguna. Pada tahap training, model ini dilatih menggunakan data yang telah dipersiapkan sebelumnya yang terdiri dari teks masukan (`text_input_prep`) dan label intents (`intents`). 
<br/><br>
Proses prediksi dimulai dengan pengguna memasukkan sebuah string teks, kemudian teks tersebut diproses dengan langkah prapemrosesan yang termasuk dalam fungsi `preprocess_text`. Setelah itu, teks yang telah diproses digunakan untuk melakukan prediksi intent menggunakan model pipeline yang telah dilatih sebelumnya. Hasil prediksi berupa probabilitas untuk setiap kelas intent, yang selanjutnya diambil nilai probabilitas tertingginya untuk menentukan kelas intent yang paling mungkin. Pada evaluasi model, metrik yang umum digunakan untuk model klasifikasi seperti ini adalah akurasi. Namun, dalam skrip ini, nilai akurasi tidak langsung dievaluasi atau dilaporkan. Model ini secara implisit melakukan evaluasi kualitas prediksi berdasarkan threshold probabilitas (di sini diatur sebesar 0.20). Jika probabilitas tertinggi dari prediksi kurang dari threshold ini, chatbot akan mengirimkan pesan bahwa tidak dapat memahami pertanyaan pengguna. Ini mengimplikasikan bahwa model dapat memiliki performa yang bervariasi tergantung pada kompleksitas dan keragaman data yang digunakan dalam training. Untuk menilai "kebagusan" atau keunggulan model ini, perlu dilakukan evaluasi lebih lanjut dengan menggunakan metrik seperti precision, recall, dan F1-score untuk masing-masing kelas intent.

<hr>

Kesimpulan:
<br/><br/>
![Algoritma](Chatbot\Documentation\image_037.png)

Model Building & Training:
<br/><br/>
![Algoritma](Chatbot\Documentation\image_018.png)

Model Evaluasi Result:
<br/><br/>
![Algoritma](Chatbot\Documentation\image_023.png)

## E. Prototype
Berikut gambaran prototype mengenai kedua AI Features kami, yang pertama adalah Chatbot:
<br/><br/>
![Proto](Rndm_Readme_Imgs\FlowChart%20ChatBot-2024-05-07-151228.png)
<br/><br/>
Flowchart tersebut menggambarkan alur interaksi antara pengguna dengan chatbot pada suatu sistem. Proses dimulai ketika pengguna berada di halaman utama dan mengklik ikon chatbot untuk memulai interaksi. Pengguna kemudian memasukkan pertanyaan atau permintaan mereka ke dalam chatbot. Sistem akan mengkategorikan pertanyaan tersebut menjadi beberapa jenis query, antara lain pertanyaan umum, pertanyaan terkait pemesanan, atau pertanyaan lainnya terkait customer service. Jika pertanyaan pengguna bersifat umum, chatbot akan mengarahkan pengguna ke bagian FAQ & QNA untuk memberikan jawaban berdasarkan Frequently Asked Questions (FAQ) atau Question and Answer (QNA) yang tersedia. Jika pertanyaan terkait dengan pemesanan, chatbot akan mencari informasi yang relevan dalam database sistem dan memberikan jawaban yang sesuai. Sedangkan jika pertanyaan pengguna bersifat lain atau terkait customer service, chatbot akan langsung memberikan jawaban. Setelah menerima jawaban, pengguna memiliki opsi untuk melanjutkan chat dengan memasukkan pertanyaan atau permintaan baru. Jika pengguna ingin melanjutkan, alur akan kembali ke langkah memasukkan pertanyaan. Jika tidak, sesi chat akan diakhiri dan sistem akan menuju ke langkah akhir untuk mengakhiri interaksi. Flowchart tersebut memberikan panduan jelas tentang bagaimana chatbot berinteraksi dengan pengguna mulai dari permulaan hingga akhir sesi.
<hr>
Berikut adalah Proto FaceRec AI kami:
<br/><br/>

![Proto](Rndm_Readme_Imgs/Flowchart%20FaceRec%20Phase1-2024-05-07-172214.png)
![Proto](Rndm_Readme_Imgs\Flowchart%20FaceRec%20Phase2-2024-05-08-014316.png)
<br/><br/>
Fase pertama dalam proses ini adalah fase pengunggahan foto atau login sebelum melakukan pemesanan. Pada tahap ini, pengguna diharuskan untuk mengunggah foto diri atau masuk ke dalam sistem, memastikan identitas mereka tercatat dengan benar. Fase ini bertujuan untuk mengumpulkan data visual yang diperlukan untuk verifikasi lebih lanjut. Fase kedua melibatkan pengambilan foto di lokasi pemesanan (on-site) untuk memastikan kesesuaian dengan data pengguna yang sudah terdaftar. Hal ini dilakukan untuk memastikan bahwa individu yang datang ke lokasi adalah benar-benar orang yang telah melakukan pemesanan. Dengan cara ini, kami dapat mencegah berbagai jenis penipuan atau scam yang mungkin terjadi, seperti pemesanan palsu atau penyalahgunaan identitas. Teknologi Face Recognition (FaceRec) digunakan dalam fitur verifikasi ini untuk membandingkan foto yang diunggah saat login dengan foto yang diambil di lokasi. Jika kedua foto tersebut cocok, pengguna akan diizinkan untuk melanjutkan ke proses berikutnya.
<hr>
Untuk Prototype cara kerja integrasinya nanti maka sama pada klasifikasi pada umumnya, berikut flowchart klasifikasi gambar PADA UMUMNYA BEGINI BERLAKU UNTUK SEMUA KLASIFIKASI DENGAN CARA DEPLOYMENT YANG SAMA
<br/><br/>

![Proto](Chatbot\Documentation\image_015.png)

## F. Deployment & Integration

### 1. Tahap 1:
- Pastikan akun DockerHub sudah dibuat
- Klik bagian 'Repositories' lalu buat sebuah repositori kosong baru
- Buat nama repositori sesuka hati, nama ini akan dibuat sama dengan nama Docker image nantinya, biarkan _visibility_ di 'Public'.
- Instal lalu buka aplikasi Docker Desktop anda.
- Pastikan Docker Machine sudah dalam status hijau dan 'running' (ada di pojok kiri bawah).
- Silakan login ke akun DockerHub anda di aplikasi Docker Desktop. (Tombol putih pojok kanan atas).
- Anda akan dibawa ke browser, dan ikuti saja step nya.
- Jika anda sudah login ke Aplikasi Docker, akun anda akan berada di pojok kanan atas. Lanjutkan ke **tahap 2.**

### 2.  Tahap 2:
- Buka project Flask API anda.
- Buka terminal anda, dan jalankan proyek Flask API anda dengan 
```python
python -m flask run
```
Atau
```python
python3 -m flask run
```
Atau
```python
py -m flask run
```
Atau pad direktori
```python
python app.py / python3 app.py / py app.py
```
- Jika berjalan dengan aman di alamat ```http://127.0.0.1:5000``` atau ```http://localhost:5000``` dengan port 5000 (yang merupakan default), maka anda bisa lanjut ke **tahap 3**. Jika belum, silakan perbaiki hingga aplikasi Flask anda berjalan dengan baik.

### 3. Tahap 3:
- Pergi ke Visual Studio Code dan buka projek Flask API anda, dan buatlah sebuah ``` Dockerfile ```.
Proyek ini akan menggunakan docker base image dari python3.10-slim
```python
FROM python:3.10-slim
```
Untuk isi dari file ini, silakan sesuaikan dengan kebutuhan anda, terutama untuk pemasangan dependency yang ada pada ``` requirements.txt ```.  Sisanya, tidak perlu diubah banyak, silakan ikuti format diatas saja.
- Kembali ke terminal, lalu buat Docker Image dengan perintah berikut :
```
docker build --tag flask-docker .
```
``` flask-docker ``` akan menjadi nama dari Docker Image anda.
- Setelahnya, tunggu proses pembuatan Docker Image, yang akan memakan waktu hingga 10-20 menit dan menggunakan ruang sebanyak 4-5 GB.
- Cek image yang tersedia dengan perintah : 
```
docker images
```
```flask-docker``` adalah nama dari Docker Image yang berhasil dibuat dari proyek Flask API sebelumnya. Disesuaikan dengan nama repositori yang sebelumnya dibuat di DockerHub.
- Jalankan Docker Image menggunakan port 5000 untuk browser dari sistem host (mode detached) dengan perintah berikut: 
```
docker run -d -p 5000:5000 flask-docker
```
```-d``` untuk mengaktifkan image dalam mode detached. <br />
```-p``` untuk menspesifikkan nomor port. Kita akan menggunakan port 5000 yang merupakan default dari Flask.
Image yang sedang berjalan dapat di cek menggunakan command:
```
docker ps
```
- Cek apakah Flask API didalam kontainer Docker Image sudah bisa diakses atau belum di port 5000.
- Melalui browser, buka ```http://localhost:5000```.
- Pastikan proses diatas sudah berhasil lancar, jika sudah, matikan container Docker Image yang berjalan, pada kasus ini, untuk nama container yang dipakai adalah ```vigorous_yonath``` (dapat di cek di kolom NAMES ketika menjalankan perintah ```docker ps```, **setiap membuat image, nama container yang dipakai akan berbeda setiap perangkat**) ini merupakan nama random yang dibuat secara otomatis ketika pembuatan docker image. Matikan container dengan perintah : 
```
docker stop vigorous_yonath
```
Lalu cek status di 
```
docker ps
```
Jika sudah kosong maka container yang menjalankan Docker Image ```flask-docker``` telah nonaktif.
- Lakukan pemberian 'tag' pada Docker Image sebelum  melakukan push ke repositori.
Berikan tag ke Docker Image dengan nama ```latest```, nama ini boleh diganti asal sesuai dengan yang di repositori.
```
docker tag flask-docker:latest azzaxsz/flask-docker:latest
```
Bagian ```azzaxsz/flask-docker:latest``` merupakan nama repositori dengan tag terbaru nya yaitu ```latest```.
- Push Docker Image ke repositori yang sudah dibuat sebelumnya di Docker Hub dengan perintah berikut : 
```
docker push azzaxsz/flask-docker:latest
```
Setelah melakukan proses tag, proses push akan memakan waktu yang cukup lama, sekitar 15-30 menit, tergantung ukuran image, untuk kasus ini, ukuran image hampir mencapai 4 GB, maka kecepatan internet akan mempengaruhi waktu proses push ke repositori.
- Ketika proses push telah selesai, maka hasil push dapat diakses di repositori yang ada di Docker Hub.
- Pada Docker Hub, terdapat tag ```latest``` yang merupakan image yang telah di push. Setelah ini lanjut ke **tahap 4**.

<hr>

![Deploy](Chatbot\Documentation\image_013.png)
![Deploy](Chatbot\Documentation\image_014.png)
![Deploy](Chatbot\Documentation\image_021.png)

<hr>

### 4. Tahap Terakhir:
- Setelah ini, kita akan melalukan deployment ke Code Engine, buka IBM Cloud dan akses layanan Code Engine.
- Klik tombol biru 'Let's Go!'.
- Pada halaman ini, silakan tekan tombol biru 'Create Project +'
- Akan muncul tab di kanan untuk membuat project baru yang akan digunakan untuk menerima Docker Image yang sebelumnya di push ke Docker Hub.
- Pastikan :
Location : ```Dallas (us-south)```.
Name : Sesuka hati kalian, dalam kasus ini : ```project-1```.
Resource Group : ```Default``` (Jika tertulis 'No resource group available' silakan hubungi mentor untuk aktivasi akun. Atau tukar akun anda.)
Tags : Biarkan kosong.
- Lalu tekan tombol biru 'Create Project'.
- Setelahnya, akan ada pilihan project di halaman pembuatan Application pada Code Engine, silakan pilih bagian 'Application'.
- Scroll ke bawah, lalu pilih nama aplikasi, dalam kasus ini, akan digunakan nama ```flask-docker```.
- Pada bagian 'Code' , tekan tombol 'Configure Image' yang ada di kanan. Biarkan pilihan di **'Use an existing container image'.**
- Akan muncul tab di kanan, dan pilih 'registry server' yang bernama ```https://index.docker.io/v1```.
- Untuk 'Registry secret' , pilih ```Create registry secret```.
- Untuk pembuatan registry secret, isi sebagai berikut : 
Secret name : bebas, pada kasus ini : ```code-engine```.
Secret contents : ```Docker Hub```.
Registry server : Biarkan begitu saja, sebelumnya sudah dipilih ```https://index.docker.io/v1```.
Username : Username Docker Hub anda. Dalam kasus ini : ```azzaxsz```.
Access Token : Silakan ambil access token anda di akun Docker Hub anda.
- Pergi ke Docker Hub, lalu ke bagian 'Account Settings'
- Pergi ke tab 'Security'
- Lalu, tekan tombol biru, 'New Access Token'
- Isi ```Access Token Description``` dengan nama sama dengan di IBM Cloud yaitu ```code-engine```, Access permissions dibiarkan saja di **Read, Write, Delete.** Lalu tekan tombol biru 'Generate'.
- Copy token tersebut dan paste di IBM Cloud Code Engine, lalu klik tombol biru 'Create'.
- Setelah membuat registry secret yang sesuai dengan akun Docker Hub, maka bagian namespace, image name dan tags akan terisi dengan sendirinya. Klik tombol biru 'Done'.
- Setelahnya, pada bagian 'Code', bagian 'Image reference' akan berisi link repositori Docker Hub anda.
- Scroll kebawah, pada bagian 'Resource and scaling'.
CPU dan memory : sesuai kebutuhan, pada kasus ini, yang akan digunakan : ``` 2vCPU / 8GB ```.
Ephemeral Storage : sesuai kebutuhan, pada kasus ini, yang akan digunakan : ``` 1,04 GB ```.
Min number of instance : 1 (jangan lebih dari 1, itu sudah cukup untuk mempertahankan API tetap hidup walau tidak dipakai).
Max number of instance : Biarkan di angka 10.
- Scroll ke paling bawah, bagian 'Image start options'.
- Ubah **listening port dari 8080 ke ```5000```**.

<hr>

![Deploy](Chatbot\Documentation\image_027.png)
![Deploy](Chatbot\Documentation\image_028.png)
![Deploy](Chatbot\Documentation\image_029.png)

<hr>

### 5. Tahap Endpoint:
- Konfigurasi lainnya tidak perlu diubah, biarkan sedia kala. Sekarang tekan tombol biru 'Create' di tab kanan : 
Sekarang, tunggu proses deployment selesai, bisa memakan waktu 5-10 menit.
- Proses deployment sudah selesai, ambil link utama untuk dilakukan pengetesan, buka bagian 'Domain mappings'
- Link public tersebut adalah yang akan digunakan untuk melakukan testing. Disebut juga sebagai API Endpoint.
- Menyesuaikan dengan yang sebelumnya di local yaitu ```http://localhost:5000``` untuk route utama (/) dan ```http://localhost:5000/predict``` untuk route prediksi. Maka untuk kasus kali, berikut ini adalah endpoint API dari Flask API yang sudah ter-deploy : 
- Route utama (/) : 
```
https://flask-docker.1igtm4p88ry2.us-south.codeengine.appdomain.cloud
```
- Route prediksi/klasifikasi intent/chatbot (/chatbot) :
```
https://flask-docker.1igtm4p88ry2.us-south.codeengine.appdomain.cloud/chatbot 
```

## G. Integrasi & Evaluasi

Dalam proyek ini, IBM Cloud Code Engine digunakan sebagai platform untuk mendeploy model Machine Learning atau Deep Learning secara online. Code Engine menyediakan lingkungan yang terkelola dengan baik untuk menjalankan kontainer yang dapat mengeksekusi model kecerdasan buatan (AI), memungkinkan aplikasi web untuk mengakses model ini melalui API terintegrasi. Setelah model ML/DL didaftarkan dan di-deploy menggunakan IBM Cloud Code Engine, pengguna dapat mengaksesnya melalui API yang disediakan. Aplikasi web dapat mengirimkan permintaan HTTP ke API ini untuk melakukan inferensi berdasarkan teks yang diinput oleh pengguna. API ini bertindak sebagai jembatan antara aplikasi frontend seperti aplikasi web dengan backend yang mengandung model ML/DL. Penggunaan model yang telah di-deploy pada aplikasi web melibatkan beberapa langkah. Pertama, aplikasi web menyediakan antarmuka pengguna (UI) yang memungkinkan pengguna untuk menginput teks. Setelah gambar diunggah, aplikasi web mengirimkan permintaan ke API model menggunakan metode POST. Permintaan ini mengandung teks yang diinput dalam format yang sesuai. Selanjutnya, API model menerima permintaan ini, menerima messages dan preprocess chat, dan menjalankan model yang telah di-deploy untuk melakukan prediksi atau klasifikasi teks tersebut untuk menentukan intent dengan tag yang tepat. Hasil prediksi dikirimkan kembali ke aplikasi web melalui respons HTTP dari API. Aplikasi web kemudian menampilkan hasil prediksi kepada pengguna, memberikan informasi tentang apa yang telah diidentifikasi oleh model, misalnya Hello/Hi maka akan dibalaskan dengan respon dataset yang ada. Integrasi ini memastikan bahwa aplikasi web dapat memanfaatkan kemampuan model AI secara efisien tanpa perlu memuat model di dalam aplikasi itu sendiri. Hal ini mengurangi overhead aplikasi dan memastikan konsistensi serta pemeliharaan yang baik terhadap model. Dengan menggunakan IBM Cloud Code Engine, deployment dan skalabilitas model dikelola secara otomatis, sehingga tim dapat lebih fokus pada pengembangan aplikasi dan meningkatkan pengalaman pengguna.
<br/><br/>
Integrasi model Machine Learning atau Deep Learning ke dalam sebuah aplikasi web berbasis React.js dengan Vite memanfaatkan IBM Cloud Code Engine sebagai platform deployment memberikan solusi yang efisien dan skalabel. Dalam konteks ini, aplikasi web yang dikembangkan menggunakan React.js dan Vite menyediakan antarmuka pengguna yang responsif dan interaktif. Pengguna dapat memasukkan teks melalui antarmuka tersebut. Setelah teks dimasukkan, aplikasi web akan mengirimkan permintaan HTTP POST ke API yang dideploy pada IBM Cloud Code Engine. API ini telah terhubung dengan model ML/DL yang telah di-training sebelumnya untuk melakukan analisis teks dan mengidentifikasi intent yang tepat berdasarkan model yang dijalankan. Proses ini termasuk preprocessing teks dan penggunaan model untuk inferensi, yang kemudian menghasilkan prediksi yang dikirimkan kembali ke aplikasi web melalui respons HTTP. Hasil prediksi ini kemudian ditampilkan kembali kepada pengguna melalui antarmuka pengguna React.js, memberikan informasi tentang apa yang telah diidentifikasi oleh model. Dengan menggunakan IBM Cloud Code Engine, tim pengembang dapat fokus pada pengembangan fitur aplikasi dan meningkatkan pengalaman pengguna tanpa harus khawatir tentang manajemen infrastruktur yang kompleks, karena deployment dan skalabilitas model dikelola secara otomatis oleh platform tersebut. Ini memastikan bahwa aplikasi web dapat memanfaatkan kecerdasan buatan secara efisien dan konsisten, meningkatkan nilai tambah aplikasi tanpa meningkatkan kompleksitas implementasi. Untuk FaceRecog kami integrasikan secara lokal atau bahasa lebih tepatnya secara manual/satu projek agar mempermudah dikarenakan mengintegrasikan sebuah library python saja dan yang menggunakan pytoml.project based itu tidak bisa ditampung di Docker.

<hr>

![Deploy](Chatbot\Documentation\image_023.png)
![Deploy](Chatbot\Documentation\image_024.png)
![Deploy](Chatbot\Documentation\image_025.png)

<hr>

## H. Conclusion

Proyek KuyFutsal telah berhasil mencapai sejumlah tujuan penting yang ditetapkan sejak awal. Dengan memanfaatkan teknologi AI dan platform web, KuyFutsal memberikan solusi inovatif untuk membantu mahasiswa mencapai kebugaran dan kesehatan yang optimal, termasuk fitur AI yang merekomendasikan makanan dan pola hidup sehat sesuai kebutuhan individu. Implementasi chatbot untuk layanan pelanggan dan FAQ di website KuyFutsal menggunakan model Multinomial Naive Bayes dari Scikit-learn telah meningkatkan efisiensi interaksi dengan pelanggan, mengurangi keterlambatan respons, dan memberikan jawaban cepat atas pertanyaan yang sering diajukan, sehingga meningkatkan pengalaman pengguna. Selain itu, integrasi teknologi pengenalan wajah untuk sistem pembayaran telah berhasil mengatasi kekhawatiran keamanan terkait aktivitas penipuan, meningkatkan keamanan, dan mengurangi risiko transaksi tidak sah. Sistem ini juga memudahkan pengguna dalam menemukan informasi tentang penyewaan lapangan futsal, menyederhanakan proses pemesanan, dan meningkatkan kepuasan pelanggan. Pengembangan integrasi AI dalam proyek ini memberikan kontribusi nyata dalam bidang kecerdasan buatan, menunjukkan aplikasi praktis dalam otomatisasi layanan pelanggan dan pemrosesan bahasa alami. Penggunaan teknologi pengenalan wajah dalam sistem pembayaran menciptakan standar baru untuk keamanan transaksi di platform e-commerce, mendorong kemajuan dalam metode otentikasi biometrik. Fokus proyek untuk meningkatkan aksesibilitas informasi penyewaan lapangan futsal dan detail kontak melalui chatbot juga berkontribusi dalam meningkatkan pengalaman pengguna pada platform manajemen fasilitas olahraga. Untuk pengembangan proyek di masa depan, rencana termasuk meningkatkan kemampuan AI, ekspansi dataset, pengembangan antarmuka pengguna yang lebih intuitif, validasi data nutrisi, dan integrasi layanan tambahan dari platform seperti IBM Cloud untuk meningkatkan fungsi dan fleksibilitas KuyFutsal.

## I. Other Documentation
Integration Docs:
![Deploy](Chatbot\Documentation\image_020.png)
![Deploy](Chatbot\Documentation\image_030.png)

FaceRec Docs:
![Deploy](FaceRec\Dokumentasi\image_031.png)
![Deploy](FaceRec\Dokumentasi\image_032.png)
![Deploy](FaceRec\Dokumentasi\image_033.png)
![Deploy](FaceRec\Dokumentasi\image_034.png)

<hr>
<br/>

Drive Documentation:
[Link Drive](https://drive.google.com/drive/folders/1y78ynnjAvrvOFSDP3CRXzJyAFy_l9z3U)<br/>

---

<h3 align="center"> Infinite Learning Indonesia </h3>
