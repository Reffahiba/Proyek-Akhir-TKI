import re
import pandas as pd
import pdfplumber
import logging
import nltk
import numpy as np
from flask import Flask, request, render_template
from nltk.tokenize import word_tokenize
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
logging.getLogger("pdfminer").setLevel(logging.ERROR)
nltk.download('punkt')

# Setup tools
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def remove_non_content_lines(text):
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if not re.match(r'^\s*(gambar|tabel|referensi|daftar pustaka)', line.lower())
        ]
        return '\n'.join(cleaned_lines)

def preprocess(text):
    text = remove_non_content_lines(text)
    text = text.lower()
    text = re.sub(r'\b[\d\W]+\b', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    tokens = word_tokenize(text)
    return tokens

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])

def analyse():
    if 'files' not in request.files:
        return "Tidak ada file yang diunggah."
    
    uploaded_files = request.files.getlist('files')
    documents = {}

    for file in uploaded_files:
        if file.filename == '':
            continue
        
        try:
            with pdfplumber.open(file) as pdf:
                full_text = ''
                for page in pdf.pages:
                    full_text += page.extract_text() + ' '
                if full_text.strip():
                    documents[file.filename] = preprocess(full_text)
                else:
                    print(f"Peringatan: Tidak ada teks yang diekstrak dari {file.filename}")

        except Exception as e:
            print(f"Gagal memproses file {file.filename}: {e}")
    
    if not documents:
        return "Tidak ada dokumen yang berhasil diproses"

    # TF-IDF
    vectorizer = TfidfVectorizer()
    doc_names = list(documents.keys())

    processed_documents_as_strings = [' '.join(tokens) for tokens in documents.values()]

    tfidf_matrix = vectorizer.fit_transform(processed_documents_as_strings)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix).round(5)

    # Tampilkan sebagai DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=doc_names, columns=doc_names)
    similarity_df_html = similarity_df.to_html(classes="table table-striped table-bordered")

    similar_words_tables_html = "<p>Fungsi untuk menampilkan kata-kata mirip antar dokumen belum diimplementasikan.</p>"

    return render_template('hasil.html', # Buat file results.html
                            similarity_table=similarity_df_html,
                            similar_words_tables=similar_words_tables_html)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
        