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
