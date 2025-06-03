import re
import pandas as pd
import pdfplumber
import logging
import nltk
from itertools import combinations
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


logging.getLogger("pdfminer").setLevel(logging.ERROR)

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
    return text

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
    vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    doc_names = list(documents.keys())
    processed_documents = list(documents.values())
    tfidf_matrix = vectorizer.fit_transform(processed_documents)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix).round(5)

    # Tampilkan sebagai DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=doc_names, columns=doc_names)
    similarity_df_html = similarity_df.to_html(classes="table table-striped table-bordered")
    similar_words_tables_html = ""

    feature_names = vectorizer.get_feature_names_out()
    for i, j in combinations(range(len(doc_names)), 2):
        doc1_name = doc_names[i]
        doc2_name = doc_names[j]

        tfidf_doc1 = tfidf_matrix[i].toarray().flatten()
        tfidf_doc2 = tfidf_matrix[j].toarray().flatten()

        common_trigrams = []
        for k, feature in enumerate(feature_names):
            if tfidf_doc1[k] > 0.01 and tfidf_doc2[k] > 0.01: 
                common_trigrams.append({
                    'Trigram': feature,
                    'Skor TF-IDF Dokumen 1': round(tfidf_doc1[k], 4),
                    'Skor TF-IDF Dokumen 2': round(tfidf_doc2[k], 4)
                })
        
        common_trigrams_df = pd.DataFrame(common_trigrams)
        if not common_trigrams_df.empty:
            common_trigrams_df['Rata-rata Skor'] = (common_trigrams_df['Skor TF-IDF Dokumen 1'] + common_trigrams_df['Skor TF-IDF Dokumen 2']) / 2
            common_trigrams_df = common_trigrams_df.sort_values(by='Rata-rata Skor', ascending=False).head(10)

            similar_words_tables_html += f"<h3 class='text-xl font-semibold mt-6 mb-2'>Kemiripan antara '{doc1_name}' dan '{doc2_name}'</h3>"
            similar_words_tables_html += "<div class='overflow-x-auto rounded-lg shadow mb-4'>"
            similar_words_tables_html += common_trigrams_df.to_html(index=False, classes="table table-striped table-bordered")
            similar_words_tables_html += "</div>"
        else:
            similar_words_tables_html += f"<p class='text-gray-600 mt-6 mb-4'>Tidak ditemukan trigram signifikan yang mirip antara '{doc1_name}' dan '{doc2_name}'.</p>"

    return render_template('hasil.html', # Buat file results.html
                            similarity_table = similarity_df_html,
                            similar_words_tables = similar_words_tables_html)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
        