import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Property Recommendation System", page_icon="🏠", layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .property-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load the trained models and tokenizer"""
    try:
        model = tf.keras.models.load_model("saved_models/best_model.keras")
        with open("saved_models/tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    # Remove URLs dan email
    text = re.sub(r"http\S+|www\S+|https\S+|\S+@\S+", "", text)
    # Handle angka dengan konteks
    text = re.sub(r"(\d+)\s*(milyar|miliar|m)", r"\1000000000", text)
    text = re.sub(r"(\d+)\s*(juta|jt)", r"\1000000", text)
    # Remove special characters tapi pertahankan yang penting
    text = re.sub(r"[^\w\s+\-.,]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the property data"""
    try:
        # Baca dataset
        df = pd.read_csv("buat_nyoba.csv")

        # Hapus baris yang kosong
        df = df.dropna()

        # Inisialisasi Sastrawi
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        stopword = stopword_factory.create_stop_word_remover()

        # Pastikan semua kolom yang diperlukan ada
        required_columns = [
            "Judul_Clean",
            "Lokasi_Clean",
            "Deskripsi_Clean",
            "Keywords_Clean",
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = df[col.replace("_Clean", "")].apply(clean_text)

        # Buat clean_description dengan weighted concatenation
        df["clean_description"] = (
            (df["Judul_Clean"].astype(str) + " " + df["Judul_Clean"].astype(str))
            + " "  # Bobot 2x
            + df["Lokasi_Clean"].astype(str)
            + " "  # Bobot 1x
            + df["Deskripsi_Clean"].astype(str)
            + " "  # Bobot 1x
            + (
                df["Keywords_Clean"].astype(str)
                + " "
                + df["Keywords_Clean"].astype(str)
            )  # Bobot 2x
        )

        # Bersihkan teks gabungan
        df["clean_description"] = df["clean_description"].apply(clean_text)

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


class PropertyRAG:
    def __init__(self, df, model, tokenizer):
        self.df = df
        self.model = model
        self.tokenizer = tokenizer
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_factory = StopWordRemoverFactory()
        self.stopword = self.stopword_factory.create_stop_word_remover()

        # Initialize TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=self.stopword_factory.get_stop_words(),
        )
        # Fit and transform the clean_description column
        self.tfidf_matrix = self.tfidf.fit_transform(df["clean_description"])

    def preprocess_query(self, query):
        """Preprocess user query"""
        # Clean the query text
        cleaned_query = clean_text(query)
        # Apply Sastrawi preprocessing
        stemmed_query = self.stemmer.stem(cleaned_query)
        cleaned_query = self.stopword.remove(stemmed_query)
        return cleaned_query

    def get_cosine_similarities(self, query):
        """Calculate cosine similarities between query and properties"""
        query_vector = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        return similarities[0]

    def get_recommendations(self, user_query, top_k=5):
        """Get property recommendations based on user query"""
        try:
            # Preprocess the query
            processed_query = self.preprocess_query(user_query)

            # Get similarity scores
            similarity_scores = self.get_cosine_similarities(processed_query)

            # Get top K similar properties
            top_indices = similarity_scores.argsort()[-top_k:][::-1]

            # Create recommendations dataframe
            recommendations = self.df.iloc[top_indices].copy()
            recommendations["relevance_score"] = similarity_scores[top_indices]

            return recommendations

        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return pd.DataFrame()


def main():
    st.title("🏠 Sistem Rekomendasi Properti Yogyakarta")

    # Load models and data
    model, tokenizer = load_models()
    df = load_and_preprocess_data()

    if model is None or tokenizer is None or df is None:
        st.error(
            "Failed to load required components. Please check your data and model files."
        )
        return

    # Initialize RAG system
    rag_system = PropertyRAG(df, model, tokenizer)

    # Sidebar for statistics
    with st.sidebar:
        st.header("📊 Statistik Properti")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Properti", f"{len(df):,}")
        with col2:
            st.metric("Rata-rata Harga", f"Rp {df['Harga'].mean():,.0f}")

        st.subheader("Distribusi Lokasi")
        location_counts = df["Lokasi_Clean"].value_counts()
        fig_loc = px.pie(
            values=location_counts.values,
            names=location_counts.index,
            title="Distribusi Properti per Lokasi",
        )
        st.plotly_chart(fig_loc, use_container_width=True)

    # Main content
    st.header("🔍 Cari Properti Impian Anda")

    # User input
    user_query = st.text_area(
        "Masukkan kriteria properti yang Anda inginkan:",
        placeholder="Contoh: saya ingin rumah di sleman dengan 3 kamar tidur, 2 kamar mandi dan slot parkir 2 mobil di bawah 1.5 milyar",
        height=100,
    )

    if st.button("🔍 Cari Rekomendasi"):
        if user_query:
            with st.spinner("Mencari rekomendasi properti yang sesuai..."):
                recommendations = rag_system.get_recommendations(user_query)

                if not recommendations.empty:
                    st.success(
                        f"Ditemukan {len(recommendations)} rekomendasi properti yang sesuai!"
                    )

                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(
                                f"""
                            <div class="property-card">
                                <h3>{row['Judul']}</h3>
                                <p><strong>📍 Lokasi:</strong> {row['Lokasi']}</p>
                                <p><strong>💰 Harga:</strong> Rp {row['Harga']:,.0f}</p>
                                <p><strong>🛏️ Kamar Tidur:</strong> {row['Kamar']} | 
                                   <strong>🚽 Kamar Mandi:</strong> {row['WC']} | 
                                   <strong>🚗 Parkir:</strong> {row['Parkir']} mobil</p>
                                <p><strong>📏 Luas Tanah:</strong> {row['Luas_Tanah']} m² | 
                                   <strong>🏠 Luas Bangunan:</strong> {row['Luas_Bangunan']} m²</p>
                                <p><strong>Skor Relevansi:</strong> {row['relevance_score']:.2f}</p>
                                <a href="{row['Property_Link']}" target="_blank">🔗 Lihat Detail Properti</a>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.warning(
                        "Maaf, tidak ditemukan properti yang sesuai dengan kriteria Anda."
                    )

    # Additional visualizations
    st.header("📊 Analisis Pasar Properti")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig_price = px.histogram(
            df,
            x="Harga",
            title="Distribusi Harga Properti",
            labels={"Harga": "Harga (Rp)", "count": "Jumlah Properti"},
        )
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        # Room vs Price correlation
        fig_corr = px.scatter(
            df,
            x="Kamar",
            y="Harga",
            title="Korelasi Jumlah Kamar vs Harga",
            labels={"Kamar": "Jumlah Kamar", "Harga": "Harga (Rp)"},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Developed with ❤️ for Yogyakarta Property Seekers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
