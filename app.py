import re
import pandas as pd
import pdfplumber
import logging
import time
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
    start_time = time.time()

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
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix[0]).round(5)

    # Tampilkan sebagai DataFrame dan tambahkan class Tailwind
    similarity_df = pd.DataFrame(similarity_matrix, index=doc_names, columns=[doc_names[0]])
    similarity_df_html = similarity_df.to_html(
        classes="min-w-full divide-y divide-gray-200 text-sm text-left text-gray-700 border border-gray-300"
    )

    similar_words_tables_html = ""
    feature_names = vectorizer.get_feature_names_out()

    if len(doc_names) < 2:
        similar_words_tables_html += "<p class='text-gray-600 mt-6 mb-4'>Unggah minimal dua dokumen untuk melihat kemiripan kata.</p>"
    else:
        doc_main_name = doc_names[0]
        tfidf_doc_main = tfidf_matrix[0].toarray().flatten()

        for j in range(1, len(doc_names)):
            doc_reference_name = doc_names[j]
            tfidf_doc_reference = tfidf_matrix[j].toarray().flatten()

            common_trigrams = []
            for k, feature in enumerate(feature_names):
                if tfidf_doc_main[k] > 0.01 and tfidf_doc_reference[k] > 0.01:
                    common_trigrams.append({
                        'Trigram': feature,
                        'Skor TF-IDF Dokumen Utama': round(tfidf_doc_main[k], 4),
                        'Skor TF-IDF Dokumen Referensi': round(tfidf_doc_reference[k], 4)
                    })
            
            common_trigrams_df = pd.DataFrame(common_trigrams)
            if not common_trigrams_df.empty:
                common_trigrams_df['Rata-rata Skor'] = (
                    (common_trigrams_df['Skor TF-IDF Dokumen Utama'] + common_trigrams_df['Skor TF-IDF Dokumen Referensi']) / 2
                )
                common_trigrams_df = common_trigrams_df.sort_values(by='Rata-rata Skor', ascending=False).head(10)

                similar_words_tables_html += f"<h3 class='text-xl font-semibold mt-6 mb-2 text-gray-800'>Kemiripan antara '{doc_main_name}' dan '{doc_reference_name}'</h3>"
                similar_words_tables_html += "<div class='overflow-x-auto rounded-lg shadow mb-4'>"
                similar_words_tables_html += common_trigrams_df.to_html(
                    index=False,
                    classes="min-w-full divide-y divide-gray-200 text-sm text-left text-gray-700 border border-gray-300"
                )
                similar_words_tables_html += "</div>"
            else:
                similar_words_tables_html += f"<p class='text-gray-600 mt-6 mb-4'>Tidak ditemukan trigram signifikan yang mirip antara '{doc_main_name}' dan '{doc_reference_name}'.</p>"

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    return render_template(
        'hasil.html',
        similarity_table=similarity_df_html,
        similar_words_tables=similar_words_tables_html,
        duration=duration
    )

if __name__ == '__main__':
    app.run(debug=True, port=5500)
        