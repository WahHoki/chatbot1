import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import threading
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import speech_recognition as sr
import pyttsx3
import os

# ==========================================
# BAGIAN 1: LOGIKA CHATBOT (BACKEND)
# ==========================================
class ChatbotBrain:
    def __init__(self, csv_file):
        print("🧠 Sedang melatih otak bot...")
        try:
            self.df = pd.read_csv(csv_file)
            # Pastikan input string
            self.df['input'] = self.df['input'].astype(str)
            self.df['output'] = self.df['output'].astype(str)
            
            self.vectorizer = TfidfVectorizer(lowercase=True)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['input'])
            self.data_ready = True
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_ready = False

    def bersihkan_teks(self, teks):
        teks = teks.lower()
        teks = teks.translate(str.maketrans('', '', string.punctuation))
        return teks

    def get_response(self, pesan_user):
        if not self.data_ready:
            return "Maaf, database error."

        pesan_bersih = self.bersihkan_teks(pesan_user)
        user_vec = self.vectorizer.transform([pesan_bersih])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix)
        
        best_index = np.argmax(similarities)
        best_score = similarities[0][best_index]
        
        # Threshold kemiripan
        if best_score > 0.2:
            return self.df.iloc[best_index]['output']
        else:
            return "Maaf, saya kurang paham maksud Anda."

# ==========================================
# BAGIAN 2: GUI (FRONTEND)
# ==========================================
class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Assistant AI (Indonesia)")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")

        # --- Inisialisasi Otak ---
        self.brain = ChatbotBrain("percakapan.csv")
        
        # --- Inisialisasi Suara (TTS) ---
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 145) # Kecepatan bicara
        
        # [BARU] PENGATURAN SUARA INDONESIA OTOMATIS
        voices = self.engine.getProperty('voices')
        found_indo = False
        print("\n--- Mencari Suara Indonesia ---")
        for voice in voices:
            # Cek apakah nama voice mengandung kata "Indonesia"
            if "indonesia" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"✅ Ditemukan & Dipilih: {voice.name}")
                found_indo = True
                break
        
        if not found_indo:
            print("⚠️ Suara Indonesia tidak ditemukan di Windows Anda.")
            print("ℹ️ Menggunakan suara default (Mungkin terdengar seperti robot asing).")
            print("💡 Tips: Install 'Indonesian Language Pack' di Settings -> Time & Language -> Speech.")

        self.recognizer = sr.Recognizer()

        # --- Komponen GUI ---
        
        # 1. Judul
        self.label_title = tk.Label(root, text="Asisten Suara Cerdas", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        self.label_title.pack(pady=10)

        # 2. Area Chat
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Arial", 10))
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.config(state='disabled') # Read only
        self.tambah_teks("Bot: Halo! Tekan tombol 'Rekam' untuk berbicara.\n")

        # 3. Label Status
        self.status_label = tk.Label(root, text="Siap.", font=("Arial", 10, "italic"), bg="#f0f0f0", fg="gray")
        self.status_label.pack(pady=5)

        # 4. Tombol Rekam
        self.btn_record = tk.Button(root, text="🎤 REKAM SUARA", font=("Arial", 12, "bold"), 
                                    bg="#4CAF50", fg="white", height=2, width=20,
                                    command=self.start_listening_thread)
        self.btn_record.pack(pady=10)

        # 5. Tombol Keluar
        self.btn_exit = tk.Button(root, text="Keluar", command=root.quit, bg="#FF5722", fg="white")
        self.btn_exit.pack(pady=5)

    def tambah_teks(self, teks, tag=None):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, teks + "\n", tag)
        self.chat_area.see(tk.END) # Auto scroll ke bawah
        self.chat_area.config(state='disabled')

    def update_status(self, teks, warna="gray"):
        self.status_label.config(text=teks, fg=warna)

    def start_listening_thread(self):
        """Menjalankan proses mendengarkan di thread terpisah agar GUI tidak macet"""
        self.btn_record.config(state='disabled', bg="#cccccc") # Matikan tombol
        threading.Thread(target=self.proses_suara).start()

    def proses_suara(self):
        # 1. Mendengarkan
        teks_user = ""
        with sr.Microphone() as source:
            self.update_status("Mendengarkan...", "red")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                self.update_status("Memproses...", "blue")
                # Gunakan ID-ID untuk speech recognition
                teks_user = self.recognizer.recognize_google(audio, language='id-ID')
            except sr.WaitTimeoutError:
                self.update_status("Waktu habis. Coba lagi.", "orange")
            except sr.UnknownValueError:
                self.update_status("Suara tidak jelas.", "orange")
            except sr.RequestError:
                self.update_status("Error koneksi internet.", "red")
            except Exception as e:
                print(f"Error lain: {e}")

        # 2. Logika Jawaban
        if teks_user:
            # Tampilkan teks user
            self.tambah_teks(f"Anda: {teks_user}")
            
            # Cari jawaban
            jawaban = self.brain.get_response(teks_user)
            
            # Tampilkan & Ucapkan jawaban
            self.tambah_teks(f"Bot: {jawaban}\n")
            self.update_status("Berbicara...", "green")
            self.bicara(jawaban)
        
        # 3. Reset UI
        self.update_status("Siap.", "gray")
        self.btn_record.config(state='normal', bg="#4CAF50") # Nyalakan tombol lagi

    def bicara(self, teks):
        """Fungsi bicara harus handle engine loop dengan hati-hati"""
        try:
            self.engine.say(teks)
            self.engine.runAndWait()
        except RuntimeError:
            # Kadang error jika engine loop sudah jalan, abaikan saja
            pass

if __name__ == "__main__":
    # Cek file CSV dulu
    if not os.path.exists("percakapan.csv"):
        print("❌ Error: File 'percakapan.csv' tidak ditemukan!")
        print("⚠️ Pastikan Anda sudah menjalankan script pembuat data sebelumnya.")
    else:
        root = tk.Tk()
        app = VoiceAssistantApp(root)
        root.mainloop()