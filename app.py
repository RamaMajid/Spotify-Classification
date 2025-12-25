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
    page_title="Spotify Popularity AI | Deep Learning",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    /* Main Theme Variables */
    :root {
        --primary: #1DB954;
        --primary-dark: #1AA34A;
        --primary-light: #1ED760;
        --secondary: #191414;
        --accent: #FF6B6B;
        --accent2: #4ECDC4;
        --accent3: #45B7D1;
        --dark: #121212;
        --darker: #0A0A0A;
        --light: #F8F9FA;
        --gray: #B3B3B3;
        --gradient: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        --gradient2: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, var(--darker) 0%, var(--dark) 100%);
        color: var(--light);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 10px rgba(29, 185, 84, 0.3);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: var(--gray);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(25, 20, 20, 0.8);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(29, 185, 84, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(29, 185, 84, 0.4);
        border-color: var(--primary);
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.95) 0%, rgba(25, 20, 20, 0.95) 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(29, 185, 84, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s infinite linear;
    }
    
    @keyframes pulse {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Buttons */
    .stButton>button {
        background: var(--gradient);
        color: white;
        font-weight: 600;
        border-radius: 1rem;
        padding: 0.8rem 2.5rem;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.5);
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--secondary) 100%);
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, var(--dark) 0%, var(--darker) 100%);
        border-right: 1px solid rgba(29, 185, 84, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(25, 20, 20, 0.7);
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 1rem 2rem;
        border: 1px solid rgba(29, 185, 84, 0.2);
        color: var(--gray);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary-light);
        border-color: var(--primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(29, 185, 84, 0.1);
        color: var(--primary);
        border-bottom: 3px solid var(--primary);
        font-weight: 600;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(25, 20, 20, 0.8);
        border-radius: 1rem;
        overflow: hidden;
    }
    
    /* Song Details Card */
    .song-card {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.1) 0%, rgba(25, 20, 20, 0.8) 100%);
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(29, 185, 84, 0.3);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .song-card:hover {
        border-color: var(--primary);
        box-shadow: 0 10px 30px rgba(29, 185, 84, 0.2);
    }
    
    /* Model Cards */
    .model-card {
        background: rgba(25, 20, 20, 0.9);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid var(--primary);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateX(5px);
        border-left-color: var(--accent2);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(90deg, transparent 0%, rgba(29, 185, 84, 0.1) 50%, transparent 100%);
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-top: 1px solid rgba(29, 185, 84, 0.2);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-light);
    }
    
    /* Selection */
    ::selection {
        background: var(--primary);
        color: white;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent !important;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: var(--gradient);
        color: white;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 0.5rem 0.5rem 0;
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
    # Enhanced Header with Logo
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-header">🎵 Spotify Popularity AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Deep Learning System for Music Popularity Classification</div>', unsafe_allow_html=True)
        
        # Badges
        st.markdown("""
            <div style="text-align: center; margin: 1.5rem 0;">
                <span class="badge">TensorFlow</span>
                <span class="badge">PyTorch</span>
                <span class="badge">Transformer</span>
                <span class="badge">Neural Networks</span>
                <span class="badge">Machine Learning</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1DB954; font-weight: 700;">⚙️ CONTROL PANEL</h2>
                <div style="height: 3px; background: linear-gradient(90deg, transparent, #1DB954, transparent); margin: 0.5rem 0;"></div>
            </div>
        """, unsafe_allow_html=True)
        
        # File upload with enhanced styling
        st.markdown("### 📁 DATA SOURCE")
        uploaded_file = st.file_uploader(
            "Upload your Spotify dataset",
            type=['csv'],
            help="Upload a CSV file or use the default dataset"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset uploaded successfully!")
            st.balloons()
        else:
            try:
                df = pd.read_csv('Dataset/spotify_data clean.csv')
                st.info("📊 Using default dataset")
            except:
                st.error("Dataset not found! Please upload a dataset.")
                return
        
        # Model selection with cards
        st.markdown("### 🤖 AI MODELS")
        st.markdown("Select a deep learning model for prediction:")
        
        model_options = {
            "MLP (Baseline)": "Neural network with Dense layers and Dropout",
            "TabNet": "Transfer learning with attention mechanism",
            "Transformer": "Multi-head attention for complex patterns"
        }
        
        model_choice = st.selectbox(
            "Choose Model",
            list(model_options.keys()),
            help="Select the AI model for prediction",
            label_visibility="collapsed"
        )
        
        # Display model info card
        st.markdown(f"""
            <div class="model-card">
                <h4 style="color: #1DB954; margin-bottom: 0.5rem;">{model_choice}</h4>
                <p style="color: #B3B3B3; font-size: 0.9rem; margin: 0;">{model_options[model_choice]}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load models
        pipeline = load_preprocessing()
        if pipeline is None:
            st.error("Preprocessing pipeline not found!")
            return
        
        model, history = None, None
        if model_choice == "MLP (Baseline)":
            model, history = load_mlp_model()
        elif model_choice == "TabNet":
            model, history = load_tabnet_model()
        elif model_choice == "Transformer":
            model, history = load_transformer_model()
        
        if model is None:
            st.error(f"Model {model_choice} not found!")
            return
        
        st.success(f"✅ {model_choice} loaded successfully!")
        
        # Additional info
        st.markdown("---")
        st.markdown("### 📊 DATASET INFO")
        st.markdown(f"**Songs:** {len(df):,}")
        st.markdown(f"**Features:** {len(df.columns)}")
        st.markdown(f"**Avg Popularity:** {df['track_popularity'].mean():.1f}")
    
    # Main content with enhanced tabs
    tab1, tab2, tab3 = st.tabs(["🎵 SONG PREDICTION", "📈 MODEL ANALYTICS", "🔍 DATA EXPLORER"])
    
    # Tab 1: Enhanced Song Prediction
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 DATASET PREVIEW")
            # Enhanced dataframe display
            styled_df = df.head(10).style.set_properties(**{
                'background-color': 'rgba(25, 20, 20, 0.8)',
                'color': '#FFFFFF',
                'border-color': '#1DB954'
            })
            st.dataframe(styled_df, use_container_width=True, height=350)
        
        with col2:
            st.markdown("### 📈 QUICK STATS")
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("""
                    <div class="metric-card">
                        <h3 style="color: #1DB954; margin-bottom: 0.5rem;">🎵</h3>
                        <h4 style="margin: 0;">{:,}</h4>
                        <p style="color: #B3B3B3; margin: 0; font-size: 0.9rem;">Total Songs</p>
                    </div>
                """.format(len(df)), unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card">
                        <h3 style="color: #4ECDC4; margin-bottom: 0.5rem;">📊</h3>
                        <h4 style="margin: 0;">{}</h4>
                        <p style="color: #B3B3B3; margin: 0; font-size: 0.9rem;">Features</p>
                    </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col_stat2:
                avg_pop = df['track_popularity'].mean()
                st.markdown("""
                    <div class="metric-card">
                        <h3 style="color: #FF6B6B; margin-bottom: 0.5rem;">🔥</h3>
                        <h4 style="margin: 0;">{:.1f}</h4>
                        <p style="color: #B3B3B3; margin: 0; font-size: 0.9rem;">Avg Popularity</p>
                    </div>
                """.format(avg_pop), unsafe_allow_html=True)
                
                duration_avg = df['track_duration_min'].mean()
                st.markdown("""
                    <div class="metric-card">
                        <h3 style="color: #45B7D1; margin-bottom: 0.5rem;">⏱️</h3>
                        <h4 style="margin: 0;">{:.1f}m</h4>
                        <p style="color: #B3B3B3; margin: 0; font-size: 0.9rem;">Avg Duration</p>
                    </div>
                """.format(duration_avg), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Song Selection
        st.markdown("### 🎤 SELECT A SONG")
        
        # Create song display with track name and artist
        df['song_display'] = df['track_name'] + " - " + df['artist_name']
        song_options = df['song_display'].tolist()
        
        selected_song = st.selectbox(
            "Choose a song for prediction:",
            song_options,
            help="Select a song from the dataset for popularity prediction",
            label_visibility="collapsed"
        )
        
        # Get selected song data
        selected_idx = song_options.index(selected_song)
        selected_data = df.iloc[selected_idx:selected_idx+1]
        
        # Display song details in enhanced card
        st.markdown("""
            <div class="song-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <h3 style="color: #1DB954; margin-bottom: 0.5rem;">🎵 {}</h3>
                        <p style="color: #B3B3B3; margin: 0.25rem 0;"><strong>Artist:</strong> {}</p>
                        <p style="color: #B3B3B3; margin: 0.25rem 0;"><strong>Album:</strong> {}</p>
                        <p style="color: #B3B3B3; margin: 0.25rem 0;"><strong>Duration:</strong> {:.2f} minutes</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.5rem; font-weight: bold; color: #1DB954;">{}</div>
                        <div style="color: #B3B3B3;">Current Popularity</div>
                    </div>
                </div>
            </div>
        """.format(
            selected_data['track_name'].values[0],
            selected_data['artist_name'].values[0],
            selected_data['album_name'].values[0],
            selected_data['track_duration_min'].values[0],
            selected_data['track_popularity'].values[0]
        ), unsafe_allow_html=True)
        
        # Predict button with enhanced styling
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            if st.button("🚀 PREDICT POPULARITY", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the song..."):
                    # Preprocess
                    X_scaled = preprocess_data(selected_data, pipeline)
                    
                    # Predict
                    predictions, predictions_proba = predict(model, model_choice, X_scaled)
                    
                    # Get class names
                    class_names = pipeline['label_encoder_target'].classes_
                    predicted_class = class_names[predictions[0]]
                    confidence = predictions_proba[0][predictions[0]]
                    
                    # Enhanced prediction display
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style="margin-bottom: 1rem; font-size: 1.2rem;">AI PREDICTION RESULT</h3>
                            <h1 style="font-size: 4rem; margin: 1rem 0; font-weight: 900;">{predicted_class}</h1>
                            <div style="font-size: 1.5rem; margin: 1rem 0;">
                                Confidence: <span style="color: #FF6B6B; font-weight: 700;">{confidence:.2%}</span>
                            </div>
                            <p style="color: rgba(255, 255, 255, 0.8); margin-top: 1rem;">
                                Powered by {model_choice}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution with enhanced chart
                    st.markdown("### 📊 CONFIDENCE DISTRIBUTION")
                    
                    # Create gauge chart for main prediction
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        title={'text': f"{predicted_class} Confidence"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#1DB954"},
                            'steps': [
                                {'range': [0, 33], 'color': "rgba(255, 107, 107, 0.3)"},
                                {'range': [33, 66], 'color': "rgba(78, 205, 196, 0.3)"},
                                {'range': [66, 100], 'color': "rgba(29, 185, 84, 0.3)"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence * 100
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=300, margin=dict(t=50, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Enhanced probability bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=class_names,
                            y=predictions_proba[0],
                            marker_color=['#1DB954' if i == predictions[0] else '#333333' 
                                         for i in range(len(class_names))],
                            text=[f'{p:.2%}' for p in predictions_proba[0]],
                            textposition='outside',
                            textfont=dict(color='white', size=12),
                            marker_line=dict(color='rgba(255,255,255,0.2)', width=1)
                        )
                    ])
                    
                    fig.update_layout(
                        title="Probability Distribution Across Classes",
                        xaxis_title="Popularity Classes",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Enhanced Model Analytics
    with tab2:
        st.markdown(f"### 📊 {model_choice.upper()} PERFORMANCE DASHBOARD")
        
        # Load evaluation results
        eval_results = load_evaluation_results()
        
        if eval_results:
            # Enhanced model comparison
            st.markdown("### 🏆 MODEL COMPARISON")
            comparison_df = pd.DataFrame(eval_results['comparison_table'])
            
            # Display as enhanced table
            st.dataframe(
                comparison_df.style
                .background_gradient(subset=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'], 
                                   cmap='Greens')
                .set_properties(**{'background-color': 'rgba(25, 20, 20, 0.8)', 'color': 'white'}),
                use_container_width=True
            )
            
            # Enhanced metrics visualization
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
            colors = ['#1DB954', '#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for metric, color in zip(metrics, colors):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].apply(lambda x: f'{x:.4f}'),
                    textposition='auto',
                    marker_color=color,
                    marker_line=dict(color='rgba(255,255,255,0.3)', width=1)
                ))
            
            fig.update_layout(
                title="Model Performance Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                barmode='group',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(
                    bgcolor='rgba(25, 20, 20, 0.8)',
                    bordercolor='rgba(29, 185, 84, 0.3)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Training history
            st.markdown("### 📉 TRAINING HISTORY")
            
            if history:
                if 'loss' in history and 'val_loss' in history:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=history['loss'],
                            mode='lines+markers',
                            name='Training Loss',
                            line=dict(color='#FF6B6B', width=3),
                            marker=dict(size=6)
                        ))
                        if 'val_loss' in history:
                            fig_loss.add_trace(go.Scatter(
                                y=history['val_loss'],
                                mode='lines+markers',
                                name='Validation Loss',
                                line=dict(color='#4ECDC4', width=3),
                                marker=dict(size=6)
                            ))
                        fig_loss.update_layout(
                            title="Training & Validation Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        fig_acc = go.Figure()
                        if 'accuracy' in history:
                            fig_acc.add_trace(go.Scatter(
                                y=history['accuracy'],
                                mode='lines+markers',
                                name='Training Accuracy',
                                line=dict(color='#1DB954', width=3),
                                marker=dict(size=6)
                            ))
                        if 'val_accuracy' in history:
                            fig_acc.add_trace(go.Scatter(
                                y=history['val_accuracy'],
                                mode='lines+markers',
                                name='Validation Accuracy',
                                line=dict(color='#45B7D1', width=3),
                                marker=dict(size=6)
                            ))
                        fig_acc.update_layout(
                            title="Training & Validation Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                
                elif 'val_accuracy' in history:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history['val_accuracy'],
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='#1DB954', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(29, 185, 84, 0.2)',
                        marker=dict(size=8, color='#1DB954')
                    ))
                    fig.update_layout(
                        title="Validation Accuracy Progress",
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Confusion Matrix
            st.markdown("### 🎯 CONFUSION MATRIX")
            
            model_key = model_choice.lower().split()[0]
            if model_key in eval_results:
                cm = np.array(eval_results[model_key]['confusion_matrix'])
                class_names = pipeline['label_encoder_target'].classes_
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale='Greens',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12, "color": "white"},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f'{model_choice} - Confusion Matrix',
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Evaluation results are not available. Please run the notebook first.")
    
    # Tab 3: Enhanced Data Explorer
    with tab3:
        st.markdown("### 🔍 ADVANCED DATA EXPLORATION")
        
        # Popularity distribution with enhanced visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 POPULARITY DISTRIBUTION")
            fig = px.histogram(df, x='track_popularity', nbins=50,
                             title='Track Popularity Distribution',
                             labels={'track_popularity': 'Popularity Score', 'count': 'Frequency'},
                             color_discrete_sequence=['#1DB954'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 POPULARITY CLASS DISTRIBUTION")
            
            def categorize_popularity(score):
                if score <= 33:
                    return 'Low'
                elif score <= 66:
                    return 'Medium'
                else:
                    return 'High'
            
            df['popularity_class'] = df['track_popularity'].apply(categorize_popularity)
            class_counts = df['popularity_class'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.5,
                marker_colors=['#FF6B6B', '#4ECDC4', '#1DB954'],
                textinfo='label+percent',
                textfont=dict(color='white', size=12)
            )])
            fig.update_layout(
                title="Popularity Class Distribution",
                height=400,
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(25, 20, 20, 0.8)',
                    bordercolor='rgba(29, 185, 84, 0.3)',
                    font=dict(color='white')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for TabNet
        if model_choice == "TabNet":
            st.markdown("### ⭐ TABNET FEATURE IMPORTANCE")
            eval_results = load_evaluation_results()
            if eval_results and 'tabnet' in eval_results:
                importance_data = eval_results['tabnet']['feature_importance']
                importance_df = pd.DataFrame(importance_data)
                
                fig = go.Figure(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#1DB954',
                    marker_line=dict(color='rgba(255,255,255,0.3)', width=1),
                    text=importance_df['importance'].apply(lambda x: f'{x:.3f}'),
                    textposition='outside',
                    textfont=dict(color='white')
                ))
                fig.update_layout(
                    title="TabNet Feature Importance Analysis",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                    💡 **TabNet's Attention Mechanism** automatically selects the most relevant features for prediction. 
                    Higher importance scores indicate features that are more critical in determining song popularity.
                """)
        
        # Enhanced Correlation Analysis
        st.markdown("### 🔗 FEATURE CORRELATION ANALYSIS")
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
            textfont={"size": 11, "color": "white"},
            hoverongaps=False
        ))
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=550,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 📋 KEY INSIGHTS")
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #1DB954;">🎵 Track Features</h4>
                    <p style="color: #B3B3B3; font-size: 0.9rem;">
                        Track number and duration show moderate correlation with popularity.
                        Explicit content may influence listener engagement.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #4ECDC4;">👨‍🎤 Artist Impact</h4>
                    <p style="color: #B3B3B3; font-size: 0.9rem;">
                        Artist popularity and follower count are strong predictors.
                        Established artists tend to have consistently popular tracks.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_insight3:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #FF6B6B;">💿 Album Context</h4>
                    <p style="color: #B3B3B3; font-size: 0.9rem;">
                        Album type and total tracks provide contextual information.
                        Single releases often differ in popularity patterns from albums.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("""
        <div class="footer">
            <h3 style="color: #1DB954; margin-bottom: 1rem;">🎓 SPOTIFY POPULARITY AI</h3>
            <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                Advanced Deep Learning System for Music Popularity Classification
            </p>
            <p style="color: #666; font-size: 0.9rem;">
                Built with ❤️ using Streamlit, TensorFlow, PyTorch, and Plotly | 
                UAP Machine Learning Project
            </p>
            <div style="margin-top: 1rem;">
                <span style="color: #1DB954; margin: 0 0.5rem;">•</span>
                <span style="color: #B3B3B3;">Real-time Predictions</span>
                <span style="color: #1DB954; margin: 0 0.5rem;">•</span>
                <span style="color: #B3B3B3;">Multiple AI Models</span>
                <span style="color: #1DB954; margin: 0 0.5ram;">•</span>
                <span style="color: #B3B3B3;">Interactive Analytics</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()