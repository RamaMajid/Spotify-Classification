import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch for TabNet
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# Sklearn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Klasifikasi Popularitas Lagu Spotify",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

# Define TransformerBlock custom layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# Load preprocessing pipeline
@st.cache_resource
def load_preprocessing():
    try:
        pipeline = joblib.load('models/preprocessing_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None

# Load models
@st.cache_resource
def load_mlp_model():
    try:
        model = keras.models.load_model('models/mlp_model.h5')
        with open('models/mlp_history.json', 'r') as f:
            history = json.load(f)
        return model, history
    except Exception as e:
        st.error(f"Error loading MLP model: {e}")
        return None, None

@st.cache_resource
def load_tabnet_model():
    try:
        model = TabNetClassifier()
        model.load_model('models/tabnet_model.zip')
        with open('models/tabnet_history.json', 'r') as f:
            history = json.load(f)
        return model, history
    except Exception as e:
        st.error(f"Error loading TabNet model: {e}")
        return None, None

@st.cache_resource
def load_transformer_model():
    try:
        # Load model with custom objects
        model = keras.models.load_model(
            'models/transformer_model.h5',
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        with open('models/transformer_history.json', 'r') as f:
            history = json.load(f)
        return model, history
    except Exception as e:
        st.error(f"Error loading Transformer model: {e}")
        return None, None

# Load evaluation results
@st.cache_data
def load_evaluation_results():
    try:
        with open('models/evaluation_results.json', 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.warning(f"Could not load evaluation results: {e}")
        return None

# Preprocess input data
def preprocess_data(df, pipeline):
    """Preprocess input data using saved pipeline"""
    # Select features
    features_to_use = ['track_number', 'track_popularity', 'explicit', 
                       'artist_popularity', 'artist_followers', 
                       'album_total_tracks', 'album_type', 'track_duration_min']
    
    X = df[features_to_use].copy()
    
    # Handle missing values
    X['track_duration_min'].fillna(X['track_duration_min'].median(), inplace=True)
    X['album_type'].fillna('unknown', inplace=True)
    
    # Encode categorical features
    X['explicit'] = X['explicit'].astype(int)
    X['album_type_encoded'] = pipeline['label_encoder_album'].transform(X['album_type'])
    X = X.drop('album_type', axis=1)
    
    # Scale features
    X_scaled = pipeline['scaler'].transform(X)
    
    return X_scaled

# Make predictions
def predict(model, model_type, X_scaled):
    """Make predictions using the selected model"""
    if model_type == "MLP (Baseline)" or model_type == "Transformer":
        predictions_proba = model.predict(X_scaled, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
    elif model_type == "TabNet":
        predictions = model.predict(X_scaled)
        predictions_proba = model.predict_proba(X_scaled)
    
    return predictions, predictions_proba

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🎵 Klasifikasi Popularitas Lagu Spotify</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">UAP Pembelajaran Mesin - Prediksi Popularitas dengan Deep Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.subheader("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (opsional)",
        type=['csv'],
        help="Upload dataset Spotify atau gunakan dataset default"
    )
    
    # Load dataset
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset berhasil diupload!")
    else:
        try:
            df = pd.read_csv('Dataset/spotify_data clean.csv')
            st.sidebar.info("📊 Menggunakan dataset default")
        except:
            st.error("Dataset tidak ditemukan! Silakan upload dataset.")
            return
    
    # Model selection
    st.sidebar.subheader("🤖 Pilih Model")
    model_choice = st.sidebar.selectbox(
        "Model",
        ["MLP (Baseline)", "TabNet", "Transformer"],
        help="Pilih model untuk prediksi"
    )
    
    # Load selected model
    pipeline = load_preprocessing()
    if pipeline is None:
        st.error("Preprocessing pipeline tidak ditemukan!")
        return
    
    model, history = None, None
    if model_choice == "MLP (Baseline)":
        model, history = load_mlp_model()
    elif model_choice == "TabNet":
        model, history = load_tabnet_model()
    elif model_choice == "Transformer":
        model, history = load_transformer_model()
    
    if model is None:
        st.error(f"Model {model_choice} tidak ditemukan!")
        return
    
    st.sidebar.success(f"✅ Model {model_choice} berhasil dimuat!")
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Info Model")
    if model_choice == "MLP (Baseline)":
        st.sidebar.info("**MLP**: Neural network sederhana dengan Dense layers dan Dropout regularization.")
    elif model_choice == "TabNet":
        st.sidebar.info("**TabNet**: Transfer learning dengan attention mechanism untuk feature selection otomatis.")
    elif model_choice == "Transformer":
        st.sidebar.info("**Transformer**: Multi-head attention untuk menangkap hubungan kompleks antar fitur.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Dataset & Prediksi", "📈 Performa Model", "🔍 Analisis"])
    
    # Tab 1: Dataset & Prediction
    with tab1:
        st.subheader("📊 Preview Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lagu", len(df))
        with col2:
            st.metric("Jumlah Fitur", len(df.columns))
        with col3:
            avg_popularity = df['track_popularity'].mean()
            st.metric("Rata-rata Popularitas", f"{avg_popularity:.1f}")
        
        st.markdown("---")
        
        # Song selection
        st.subheader("🎵 Pilih Lagu untuk Prediksi")
        
        # Create song display with track name and artist
        df['song_display'] = df['track_name'] + " - " + df['artist_name']
        song_options = df['song_display'].tolist()
        
        selected_song = st.selectbox(
            "Pilih lagu:",
            song_options,
            help="Pilih lagu dari dataset untuk melihat prediksi popularitas"
        )
        
        # Get selected song data
        selected_idx = song_options.index(selected_song)
        selected_data = df.iloc[selected_idx:selected_idx+1]
        
        # Display song details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎤 Detail Lagu")
            st.write(f"**Judul**: {selected_data['track_name'].values[0]}")
            st.write(f"**Artis**: {selected_data['artist_name'].values[0]}")
            st.write(f"**Album**: {selected_data['album_name'].values[0]}")
            st.write(f"**Durasi**: {selected_data['track_duration_min'].values[0]:.2f} menit")
        
        with col2:
            st.markdown("### 📊 Statistik")
            st.write(f"**Popularitas Aktual**: {selected_data['track_popularity'].values[0]}")
            st.write(f"**Popularitas Artis**: {selected_data['artist_popularity'].values[0]}")
            st.write(f"**Followers Artis**: {selected_data['artist_followers'].values[0]:,}")
            st.write(f"**Tipe Album**: {selected_data['album_type'].values[0]}")
        
        # Predict button
        if st.button("🔮 Prediksi Popularitas", use_container_width=True):
            with st.spinner("Memproses prediksi..."):
                # Preprocess
                X_scaled = preprocess_data(selected_data, pipeline)
                
                # Predict
                predictions, predictions_proba = predict(model, model_choice, X_scaled)
                
                # Get class names
                class_names = pipeline['label_encoder_target'].classes_
                predicted_class = class_names[predictions[0]]
                
                # Display prediction
                st.markdown("---")
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Hasil Prediksi</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_class}</h1>
                    <p style="font-size: 1.2rem;">Confidence: {predictions_proba[0][predictions[0]]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.subheader("📊 Distribusi Probabilitas")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=class_names,
                        y=predictions_proba[0],
                        marker_color=['#1DB954' if i == predictions[0] else '#b3b3b3' 
                                     for i in range(len(class_names))],
                        text=[f'{p:.2%}' for p in predictions_proba[0]],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Probabilitas untuk Setiap Kelas",
                    xaxis_title="Kelas Popularitas",
                    yaxis_title="Probabilitas",
                    yaxis_range=[0, 1],
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Model Performance
    with tab2:
        st.subheader(f"📈 Performa Model: {model_choice}")
        
        # Load evaluation results
        eval_results = load_evaluation_results()
        
        if eval_results:
            # Model comparison
            st.markdown("### 🏆 Perbandingan Model")
            comparison_df = pd.DataFrame(eval_results['comparison_table'])
            st.dataframe(comparison_df, use_container_width=True)
            
            # Metrics visualization
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
            
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].apply(lambda x: f'{x:.4f}'),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Perbandingan Metrik Evaluasi",
                xaxis_title="Model",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Training history
            st.markdown("### 📉 Training History")
            
            if history:
                if 'loss' in history and 'val_loss' in history:
                    # For MLP and Transformer
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#FF6B6B', width=2)
                        ))
                        if 'val_loss' in history:
                            fig_loss.add_trace(go.Scatter(
                                y=history['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#4ECDC4', width=2)
                            ))
                        fig_loss.update_layout(
                            title="Training & Validation Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        fig_acc = go.Figure()
                        if 'accuracy' in history:
                            fig_acc.add_trace(go.Scatter(
                                y=history['accuracy'],
                                mode='lines',
                                name='Training Accuracy',
                                line=dict(color='#FF6B6B', width=2)
                            ))
                        if 'val_accuracy' in history:
                            fig_acc.add_trace(go.Scatter(
                                y=history['val_accuracy'],
                                mode='lines',
                                name='Validation Accuracy',
                                line=dict(color='#4ECDC4', width=2)
                            ))
                        fig_acc.update_layout(
                            title="Training & Validation Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            height=400
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                
                elif 'val_accuracy' in history:
                    # For TabNet
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history['val_accuracy'],
                        mode='lines',
                        name='Validation Accuracy',
                        line=dict(color='#1DB954', width=2)
                    ))
                    fig.update_layout(
                        title="Validation Accuracy",
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            st.markdown("### 🎯 Confusion Matrix")
            
            model_key = model_choice.lower().split()[0]
            if model_key in eval_results:
                cm = np.array(eval_results[model_key]['confusion_matrix'])
                class_names = pipeline['label_encoder_target'].classes_
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, 
                           yticklabels=class_names,
                           ax=ax)
                ax.set_title(f'{model_choice} - Confusion Matrix', fontsize=14, fontweight='bold')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
        else:
            st.warning("Evaluation results tidak tersedia. Jalankan notebook terlebih dahulu.")
    
    # Tab 3: Analysis
    with tab3:
        st.subheader("🔍 Analisis Dataset")
        
        # Popularity distribution
        st.markdown("### 📊 Distribusi Popularitas")
        
        fig = px.histogram(df, x='track_popularity', nbins=50,
                          title='Distribusi Track Popularity',
                          labels={'track_popularity': 'Popularity Score', 'count': 'Frequency'})
        fig.update_traces(marker_color='#1DB954')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create popularity classes for visualization
        def categorize_popularity(score):
            if score <= 33:
                return 'Tidak Populer'
            elif score <= 66:
                return 'Populer'
            else:
                return 'Sangat Populer'
        
        df['popularity_class'] = df['track_popularity'].apply(categorize_popularity)
        
        # Class distribution
        st.markdown("### 🎯 Distribusi Kelas Popularitas")
        class_counts = df['popularity_class'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )])
        fig.update_layout(title="Proporsi Kelas Popularitas", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for TabNet)
        if model_choice == "TabNet":
            st.markdown("### ⭐ Feature Importance (TabNet)")
            eval_results = load_evaluation_results()
            if eval_results and 'tabnet' in eval_results:
                importance_data = eval_results['tabnet']['feature_importance']
                importance_df = pd.DataFrame(importance_data)
                
                fig = go.Figure(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#1DB954'
                ))
                fig.update_layout(
                    title="Feature Importance dari TabNet Attention Mechanism",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **TabNet Attention Mechanism** secara otomatis memilih fitur yang paling relevan untuk prediksi. Semakin tinggi importance score, semakin penting fitur tersebut dalam menentukan popularitas lagu.")
        
        # Correlation analysis
        st.markdown("### 🔗 Analisis Korelasi")
        numerical_cols = ['track_number', 'track_popularity', 'artist_popularity', 
                          'artist_followers', 'album_total_tracks', 'track_duration_min']
        
        corr_matrix = df[numerical_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
        ))
        fig.update_layout(
            title="Correlation Matrix",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎓 UAP Pembelajaran Mesin - Klasifikasi Popularitas Lagu Spotify</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit, TensorFlow, dan PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
