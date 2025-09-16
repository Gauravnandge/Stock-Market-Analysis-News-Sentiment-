import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import warnings
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# --- LANGCHAIN AND GOOGLE AI IMPORTS (RAG & GENERAL CHATBOT) ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains import ConversationChain
# RAG Components
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- NEW IMPORTS FOR DOWNLOADING ---
import gdown
import zipfile
import shutil

warnings.filterwarnings('ignore')

# --- NEW: FUNCTION TO DOWNLOAD AND UNZIP FILES FROM GOOGLE DRIVE ---
def download_and_unzip_files():
    """Downloads and unzips model files from Google Drive if they don't already exist."""
    # Check if a key directory exists to prevent re-downloading on every run
    if os.path.exists("models") and os.path.exists("sentiment_models") and os.path.exists("scalers"):
        st.toast("Model and data files already loaded.", icon="✅")
        return

    st.info("Downloading required model and data files. This may take a moment...", icon="⏳")
    progress_bar = st.progress(0, text="Starting download...")

    try:
        # Google Drive folder ID from the provided link
        folder_id = "1KlrX-8p0NpQEWgPDLm7zypHF1BHLalid"
        output_zip_path = "stock_sentiment_files.zip"

        # Use gdown to download the folder as a zip file
        progress_bar.progress(10, text="Contacting Google Drive...")
        gdown.download_folder(id=folder_id, output=output_zip_path, quiet=True, use_cookies=False)
        progress_bar.progress(50, text="Extracting files...")

        # Unzip the downloaded file
        with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
            zip_ref.extractall(".") # Extract to the current directory

        # gdown downloads the folder into a directory named after the Google Drive folder.
        # We need to move the contents out of this sub-directory.
        source_folder = "Stock sentiment" # This is the name of your folder on Google Drive
        if os.path.exists(source_folder):
             # Move all contents from the nested folder to the current app directory
            for item in os.listdir(source_folder):
                source_path = os.path.join(source_folder, item)
                dest_path = os.path.join(".", item)
                if os.path.isdir(source_path):
                    if os.path.exists(dest_path): # If destination directory exists, remove it first
                        shutil.rmtree(dest_path)
                    shutil.move(source_path, dest_path)
                else:
                    shutil.move(source_path, dest_path)
            # Remove the now-empty source folder
            shutil.rmtree(source_folder)

        # Clean up the downloaded zip file
        os.remove(output_zip_path)
        
        progress_bar.progress(100, text="Download and setup complete!")
        st.success("✅ Files downloaded and extracted successfully!")
        st.balloons()

    except Exception as e:
        st.error(f"Error downloading or unzipping files: {e}")
        st.error("The application may not function correctly. Please check the Google Drive link and permissions.")
        st.stop()

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CUSTOM CSS STYLING -----------------
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Header animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #00A0B0;
        animation: fadeIn 1s ease-in-out;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #2E2E2E;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0,160,176,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #A0A0A0;
        font-weight: bold;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-weight: bold;
    }

    /* Custom Info Cards */
    .info-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        border: 1px solid #00A0B0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        transition: transform 0.3s ease-in-out;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,160,176,0.3);
    }

    /* Section headers */
    .section-header {
        color: #00A0B0;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00A0B0;
        padding-bottom: 0.5rem;
    }

    /* Company name styling */
    .company-name {
        color: #B0B0B0;
        font-size: 0.9rem;
        font-style: italic;
        margin-left: 0.5rem;
    }

    /* News card styling */
    .news-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border-left: 4px solid #00A0B0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Sentiment Analysis Cards */
    .sentiment-card {
        background: linear-gradient(135deg, #1E3A1E 0%, #2E4A2E 100%);
        border: 1px solid #4CAF50;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(76,175,80,0.2);
    }

    .sentiment-positive {
        border-color: #4CAF50 !important;
        background: linear-gradient(135deg, #1E3A1E 0%, #2E4A2E 100%) !important;
    }

    .sentiment-negative {
        border-color: #F44336 !important;
        background: linear-gradient(135deg, #3A1E1E 0%, #4A2E2E 100%) !important;
    }

    .sentiment-neutral {
        border-color: #FFC107 !important;
        background: linear-gradient(135deg, #3A3A1E 0%, #4A4A2E 100%) !important;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .chat-user {
        background-color: #2E2E2E;
        border-left: 4px solid #00A0B0;
    }

    .chat-assistant {
        background-color: #1E2E1E;
        border-left: 4px solid #4CAF50;
    }

    /* BI Dashboard placeholder */
    .bi-placeholder {
        background: linear-gradient(45deg, #1E1E1E, #2E2E2E);
        border: 2px dashed #00A0B0;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }

    .selectbox div[data-baseweb="select"] > div {
        background-color: #2E2E2E;
        border-color: #444;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CONSTANTS & MODEL SETUP -----------------
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOGL', 'AMD', 'NFLX', 'JPM']
LOOK_BACK_PERIOD = 100  # Updated to match training code

# Company names mapping
COMPANY_NAMES = {
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc',
    'AAPL': 'Apple Inc',
    'AMZN': 'Amazon.com Inc',
    'MSFT': 'Microsoft Corporation',
    'META': 'Meta Platforms Inc',
    'GOOGL': 'Alphabet Inc',
    'AMD': 'Advanced Micro Devices',
    'NFLX': 'Netflix Inc',
    'JPM': 'JPMorgan Chase & Co'
}

# Enhanced Attention Layer (Updated from training code)
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ----------------- SENTIMENT ANALYSIS SETUP -----------------
@st.cache_resource
def load_sentiment_models():
    """Load pre-trained sentiment analysis models"""
    try:
        # --- MODIFIED: Updated file paths to match downloaded folder structure ---
        finbert_folder = 'finbert_stock_sentiment_model'
        sentiment_folder = 'sentiment_models'
        
        # Path to the FinBERT model directory
        model_path = os.path.join(sentiment_folder, finbert_folder)
        
        # Paths to the joblib files
        impact_model_path = os.path.join(sentiment_folder, 'impact_prediction_model.pkl')
        label_encoder_path = os.path.join(sentiment_folder, 'label_encoder.pkl')
        # --- END OF MODIFICATION ---
        
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Load impact prediction model and label encoder
            impact_model = joblib.load(impact_model_path)
            label_encoder = joblib.load(label_encoder_path)

            return tokenizer, model, impact_model, label_encoder, device
        else:
            st.error("Sentiment models not found. Please ensure the download was successful.")
            return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading sentiment models: {str(e)}")
        return None, None, None, None, None

def clean_headline(headline):
    """Clean headline text for sentiment analysis"""
    if not isinstance(headline, str):
        return ""
    headline = re.sub(r'\[.*?\]', '', headline)  # Remove text in square brackets
    headline = re.sub(r'http\S+', '', headline)  # Remove URLs
    headline = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', headline)  # Keep some punctuation
    headline = headline.lower()  # Convert to lowercase
    return headline.strip()

def predict_sentiment_and_impact(headline, symbol):
    """Predict sentiment and stock price impact for a given headline"""
    tokenizer, model, impact_model, label_encoder, device = load_sentiment_models()

    if not all([tokenizer, model, impact_model, label_encoder]):
        # Fallback to rule-based approach if models aren't available
        return predict_sentiment_fallback(headline, symbol)

    try:
        # Clean the headline
        cleaned_headline = clean_headline(headline)

        # Tokenize and encode
        encoding = tokenizer.encode_plus(
            cleaned_headline,
            add_special_tokens=True,
            max_length=160,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Predict sentiment
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted_class = torch.max(outputs.logits, dim=1)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get sentiment label and confidence
        predicted_sentiment = predicted_class.cpu().numpy()[0]
        sentiment_label = label_encoder.inverse_transform([predicted_sentiment])[0]
        confidence = float(torch.max(probabilities).cpu().numpy())

        # Enhanced impact calculation with realistic bounds
        raw_impact = impact_model.predict([[predicted_sentiment]])[0]

        # Apply realistic bounds and confidence weighting
        impact_percentage = calculate_realistic_impact(
            sentiment_label, confidence, raw_impact, symbol, headline
        )

        # Get current stock price to calculate predicted price
        current_price = get_current_stock_price(symbol)
        predicted_price = None
        if current_price:
            predicted_price = current_price * (1 + impact_percentage / 100)

        return sentiment_label, confidence, impact_percentage, predicted_price

    except Exception as e:
        st.error(f"Error in sentiment prediction: {str(e)}")
        return predict_sentiment_fallback(headline, symbol)

def calculate_realistic_impact(sentiment_label, confidence, raw_impact, symbol, headline):
    """
    Calculate realistic price impact using the model's output ('raw_impact')
    and applying confidence weighting and logical bounds.
    """
    sentiment_key = sentiment_label.lower()

    # Use the model's raw_impact as the base
    base_impact = raw_impact

    # Ensure the base impact aligns with the sentiment
    if sentiment_key == 'positive':
        base_impact = abs(base_impact)
    elif sentiment_key == 'negative':
        base_impact = -abs(base_impact)
    else: # Neutral
        base_impact *= 0.2 # Drastically reduce impact for neutral news

    # Weight the impact by the model's confidence
    confidence_factor = max(0.4, confidence) # Minimum 40% confidence factor
    scaled_impact = base_impact * confidence_factor

    # Keyword-based multipliers for significant events
    headline_lower = headline.lower()
    if any(word in headline_lower for word in ['earnings', 'revenue', 'profit', 'quarterly']):
        scaled_impact *= 1.5
    elif any(word in headline_lower for word in ['acquisition', 'merger', 'partnership', 'breakthrough', 'fda', 'approval']):
        scaled_impact *= 1.8
    elif any(word in headline_lower for word in ['lawsuit', 'investigation', 'recall']):
        scaled_impact *= 1.6
    elif any(word in headline_lower for word in ['market', 'economy', 'fed', 'inflation']):
        scaled_impact *= 0.6 # Market-wide news has less impact on a single stock

    # Apply final, realistic caps to prevent extreme outliers (e.g., +/- 8%)
    final_impact = max(-8.0, min(8.0, scaled_impact))

    return round(final_impact, 2)

def predict_sentiment_fallback(headline, symbol):
    """Fallback sentiment prediction using rule-based approach"""
    headline_lower = headline.lower()

    # Positive keywords
    positive_words = ['beat', 'exceed', 'growth', 'profit', 'gain', 'rise', 'surge', 'strong',
                      'excellent', 'breakthrough', 'success', 'record', 'high', 'boost', 'up']

    # Negative keywords
    negative_words = ['miss', 'fall', 'drop', 'decline', 'loss', 'weak', 'concern', 'risk',
                      'lawsuit', 'investigation', 'down', 'plunge', 'crash', 'fail', 'warning']

    positive_count = sum(1 for word in positive_words if word in headline_lower)
    negative_count = sum(1 for word in negative_words if word in headline_lower)

    if positive_count > negative_count:
        sentiment = 'positive'
        confidence = min(0.8, 0.6 + positive_count * 0.1)
        impact = np.random.uniform(0.5, 2.0)
    elif negative_count > positive_count:
        sentiment = 'negative'
        confidence = min(0.8, 0.6 + negative_count * 0.1)
        impact = np.random.uniform(-2.0, -0.5)
    else:
        sentiment = 'neutral'
        confidence = 0.6
        impact = np.random.uniform(-0.3, 0.3)

    current_price = get_current_stock_price(symbol)
    predicted_price = None
    if current_price:
        predicted_price = current_price * (1 + impact / 100)

    return sentiment, confidence, impact, predicted_price

def get_current_stock_price(symbol):
    """Get current stock price for a symbol"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception:
        pass
    return None

# ----------------- DATA FETCHING & PROCESSING FUNCTIONS -----------------
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1y"):
    """Fetch stock data from yfinance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def add_technical_indicators(df):
    """Add comprehensive technical indicators (enhanced from training code)"""
    # Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(3).mean()

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([df['High'] - df['Low'],
                    np.abs(df['High'] - df['Close'].shift(1)),
                    np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/14).mean()) / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX_14'] = dx.ewm(alpha=1/14).mean()

    # Bollinger Bands
    sma_20_bb = df['Close'].rolling(window=20).mean()
    std_dev_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20_bb + (std_dev_20 * 2)
    df['BB_Lower'] = sma_20_bb - (std_dev_20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20_bb

    # Average True Range
    df['ATR'] = atr

    # On-Balance Volume
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # Chaikin Money Flow
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfv.fillna(0)
    mfv *= df['Volume']
    df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()

    # Time-based features
    if 'Date' in df.columns:
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month_of_year'] = df['Date'].dt.month
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    else:
        df['day_of_week'] = df.index.dayofweek
        df['month_of_year'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Forward fill and drop NaN values
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=300)
def get_market_overview():
    """Get market overview with top gainers and losers"""
    gainers, losers = [], []
    market_stats = {'total_volume': 0, 'avg_change': 0, 'positive_stocks': 0}

    for symbol in SYMBOLS:
        try:
            hist = yf.Ticker(symbol).history(period="2d")
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = ((current_price - prev_price) / prev_price) * 100
                volume = hist['Volume'].iloc[-1]

                stock_data = {
                    'symbol': symbol,
                    'company': COMPANY_NAMES.get(symbol, symbol),
                    'price': current_price,
                    'change': change,
                    'volume': volume
                }

                market_stats['total_volume'] += volume
                market_stats['avg_change'] += change
                if change > 0:
                    gainers.append(stock_data)
                    market_stats['positive_stocks'] += 1
                else:
                    losers.append(stock_data)
        except Exception:
            continue

    if len(SYMBOLS) > 0:
        market_stats['avg_change'] /= len(SYMBOLS)
    gainers = sorted(gainers, key=lambda x: x['change'], reverse=True)[:3]
    losers = sorted(losers, key=lambda x: x['change'])[:3]

    return gainers, losers, market_stats

@st.cache_data(ttl=600)
def get_market_indices():
    """Get major market indices data"""
    indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
    index_data = []

    for symbol, name in indices.items():
        try:
            hist = yf.Ticker(symbol).history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                index_data.append({'name': name, 'value': current, 'change': change})
        except Exception:
            continue

    return index_data

def simulate_news_sentiment():
    """Simulate news sentiment analysis (placeholder function)"""
    news_data = []
    sentiments = ['Positive', 'Negative', 'Neutral']

    for symbol in SYMBOLS[:5]: # Show news for top 5 stocks
        sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])
        impact = np.random.uniform(0.5, 2.5)
        predicted_trend = "Bullish" if sentiment == "Positive" else "Bearish" if sentiment == "Negative" else "Neutral"

        news_data.append({
            'symbol': symbol,
            'company': COMPANY_NAMES.get(symbol, symbol),
            'sentiment': sentiment,
            'impact_score': impact,
            'predicted_trend': predicted_trend,
            'headline': f"{COMPANY_NAMES.get(symbol, symbol)} shows {sentiment.lower()} market signals"
        })

    return news_data

# ----------------- PLOTTING FUNCTIONS -----------------
def create_candlestick_chart(df, title):
    """Create candlestick chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f'{title} Price', 'Volume'), row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines',
                             name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines',
                             name='SMA 50', line=dict(color='cyan', width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                         marker_color='rgba(0,160,176,0.6)'), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        template='plotly_dark'
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ----------------- PREDICTION FUNCTIONS (UPDATED FOR NEW TRAINING SCRIPT) -----------------

def load_model_and_scaler(symbol):
    """
    Load trained model and the feature scaler for prediction.
    This is updated to load ONLY the feature scaler, as the new model
    predicts log returns and doesn't use a target scaler.
    """
    try:
        model_path = os.path.join('models', f'{symbol}_best_model.keras')
        feature_scaler_path = os.path.join('scalers', f'{symbol}_feature_scaler.pkl')

        # Check if both required files exist
        if os.path.exists(model_path) and os.path.exists(feature_scaler_path):
            custom_objects = {'Attention': Attention}
            model = load_model(model_path, custom_objects=custom_objects)
            feature_scaler = joblib.load(feature_scaler_path)
            # The target scaler is no longer needed
            return model, feature_scaler
    except Exception as e:
        st.error(f"Error loading model or scaler for {symbol}: {str(e)}")

    # Return None for both if any file is missing or an error occurs
    return None, None

def predict_next_price(symbol, df):
    """
    Predict next day's price using the new model that outputs log returns.
    """
    # Load the model and the feature scaler (no target scaler anymore)
    model, feature_scaler = load_model_and_scaler(symbol)
    if not all([model, feature_scaler]):
        st.error(f"Model or feature scaler for {symbol} is missing. Please check if the files were downloaded correctly.")
        return None, None, None

    try:
        # Feature columns must exactly match the training script
        feature_cols = [
            'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP', 'CMF',
            'ROC', 'RSI_14', '%K', '%D', 'SMA_20', 'SMA_50', 'MACD',
            'MACD_signal', 'MACD_hist', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'ATR', 'OBV', 'ADX_14', 'day_of_week', 'month_of_year', 'week_of_year'
        ]

        # Get the last sequence of data
        recent_data = df[feature_cols].tail(LOOK_BACK_PERIOD)
        if len(recent_data) < LOOK_BACK_PERIOD:
            st.warning(f"Not enough historical data for {symbol} to make a prediction (need {LOOK_BACK_PERIOD} days, have {len(recent_data)}).")
            return None, None, None

        # Get the last known closing price to calculate the final predicted price
        last_actual_price = recent_data['Close'].iloc[-1]

        # Scale features using the loaded feature scaler
        scaled_features = feature_scaler.transform(recent_data)
        X_pred = scaled_features.reshape(1, LOOK_BACK_PERIOD, len(feature_cols))

        # Make prediction
        predictions = model.predict(X_pred, verbose=0)

        # --- NEW LOGIC: Handle dictionary output and log return calculation ---

        # 1. Extract predictions from the model's dictionary output
        predicted_log_return = predictions['price'][0][0]
        predicted_direction_probs = predictions['direction'][0]

        # 2. Calculate the predicted price from the log return
        # Formula: Predicted Price = Last Actual Price * e^(log_return)
        predicted_price = last_actual_price * np.exp(predicted_log_return)

        # 3. Determine the predicted direction and confidence
        # This mapping must match your training script's target creation
        # 0: Down, 1: Sideways, 2: Up
        direction_map = {0: "Down", 1: "Sideways", 2: "Up"}
        predicted_direction_index = np.argmax(predicted_direction_probs)
        direction = direction_map.get(predicted_direction_index, "Unknown")
        direction_confidence = predicted_direction_probs[predicted_direction_index] * 100

        return predicted_price, direction, direction_confidence

    except Exception as e:
        st.error(f"An error occurred during prediction for {symbol}: {str(e)}")
        return None, None, None

# ----------------- NEWS SENTIMENT ANALYSIS PAGE -----------------
def render_news_sentiment_analysis():
    """Renders the News Sentiment Analysis page"""
    st.markdown('<p class="section-header">📰 AI News Sentiment Analysis & Price Impact</p>', unsafe_allow_html=True)

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter News Headline")
        user_headline = st.text_area(
            "Paste a financial news headline here:",
            placeholder="e.g., Apple reports record quarterly earnings beating analyst expectations",
            height=100
        )

    with col2:
        st.subheader("Select Stock Symbol")
        selected_symbol = st.selectbox(
            "Choose stock symbol:",
            SYMBOLS,
            key="sentiment_stock"
        )

        if st.button("Analyze Sentiment & Impact", type="primary"):
            if user_headline.strip():
                with st.spinner("Analyzing sentiment and calculating price impact..."):
                    sentiment, confidence, impact_pct, predicted_price = predict_sentiment_and_impact(
                        user_headline, selected_symbol
                    )

                    if sentiment is not None:
                        # Store results in session state
                        st.session_state.sentiment_results = {
                            'headline': user_headline,
                            'symbol': selected_symbol,
                            'company': COMPANY_NAMES.get(selected_symbol, selected_symbol),
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'impact_pct': impact_pct,
                            'predicted_price': predicted_price,
                            'current_price': get_current_stock_price(selected_symbol)
                        }

    # Display results
    if hasattr(st.session_state, 'sentiment_results') and st.session_state.sentiment_results:
        results = st.session_state.sentiment_results

        st.markdown("---")
        st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)

        # Results display
        col1, col2, col3 = st.columns(3)

        # Sentiment card styling based on sentiment
        sentiment_class = f"sentiment-{results['sentiment'].lower()}"
        sentiment_color = {
            'positive': '#4CAF50',
            'negative': '#F44336',
            'neutral': '#FFC107'
        }.get(results['sentiment'].lower(), '#FFC107')

        with col1:
            st.markdown(f"""
            <div class="sentiment-card {sentiment_class}">
                <h3 style="color: {sentiment_color}; margin-top: 0;">📈 Sentiment Analysis</h3>
                <p><strong>Sentiment:</strong> {results['sentiment']}</p>
                <p><strong>Confidence:</strong> {results['confidence']:.2%}</p>
                <p><strong>Stock:</strong> {results['symbol']} - {results['company']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="sentiment-card">
                <h3 style="color: #00A0B0; margin-top: 0;">💰 Price Impact</h3>
                <p><strong>Current Price:</strong> ${results['current_price']:.2f}</p>
                <p><strong>Impact:</strong> {results['impact_pct']:+.2f}%</p>
                <p><strong>Predicted Price:</strong> ${results['predicted_price']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            price_change = results['predicted_price'] - results['current_price']
            change_direction = "📈" if price_change > 0 else "📉"

            st.markdown(f"""
            <div class="sentiment-card">
                <h3 style="color: #FF9800; margin-top: 0;">🎯 Prediction Summary</h3>
                <p><strong>Price Change:</strong> {change_direction} ${price_change:+.2f}</p>
                <p><strong>Direction:</strong> {"Bullish" if price_change > 0 else "Bearish"}</p>
                <p><strong>Risk Level:</strong> {"High" if abs(results['impact_pct']) > 2 else "Medium" if abs(results['impact_pct']) > 1 else "Low"}</p>
            </div>
            """, unsafe_allow_html=True)

        # Visualization
        st.markdown('<p class="section-header">📊 Price Impact Visualization</p>', unsafe_allow_html=True)

        # Create gauge chart for sentiment confidence
        col1, col2 = st.columns(2)

        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = results['confidence'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Confidence (%)"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': sentiment_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig_gauge.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Price comparison chart
            labels = ['Current Price', 'Predicted Price']
            values = [results['current_price'], results['predicted_price']]
            colors = ['#00A0B0', sentiment_color]

            fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig_bar.update_layout(
                title="Price Comparison",
                yaxis_title="Price ($)",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Historical context
        with st.expander("📈 View Historical Context"):
            hist, _ = get_stock_data(results['symbol'], "1mo")
            if hist is not None:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#00A0B0')
                ))

                # Add predicted price point
                next_date = hist.index[-1] + timedelta(days=1)
                fig_hist.add_trace(go.Scatter(
                    x=[next_date],
                    y=[results['predicted_price']],
                    mode='markers',
                    name='Predicted Price',
                    marker=dict(size=15, color=sentiment_color, symbol='star')
                ))

                fig_hist.update_layout(
                    title=f"{results['symbol']} - Recent Price History with Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template='plotly_dark'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- PAGE RENDERING FUNCTIONS -----------------
def render_market_overview():
    """Renders the enhanced Market Overview page"""
    st.markdown('<p class="section-header">🌍 Market at a Glance</p>', unsafe_allow_html=True)

    # Market Indices
    with st.spinner("Loading market indices..."):
        indices_data = get_market_indices()

    if indices_data:
        cols = st.columns(len(indices_data))
        for i, index in enumerate(indices_data):
            with cols[i]:
                st.metric(
                    label=index['name'],
                    value=f"{index['value']:,.2f}",
                    delta=f"{index['change']:+.2f}%"
                )

    st.markdown("---")

    # Top Gainers and Losers
    with st.spinner("Loading market data..."):
        gainers, losers, market_stats = get_market_overview()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 Top 3 Gainers")
        for stock in gainers:
            st.markdown(f"""
            <div class="info-card">
                <strong>{stock['symbol']}</strong>
                <span class="company-name">{stock['company']}</span><br>
                <strong>${stock['price']:.2f}</strong>
                <span style="color: #4CAF50; font-weight: bold;">+{stock['change']:.2f}%</span><br>
                <small>Volume: {stock['volume']:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("📉 Top 3 Losers")
        for stock in losers:
            st.markdown(f"""
            <div class="info-card">
                <strong>{stock['symbol']}</strong>
                <span class="company-name">{stock['company']}</span><br>
                <strong>${stock['price']:.2f}</strong>
                <span style="color: #F44336; font-weight: bold;">{stock['change']:.2f}%</span><br>
                <small>Volume: {stock['volume']:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)

    # Market Statistics
    st.markdown('<p class="section-header">📊 Market Statistics</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Sentiment", f"{market_stats['avg_change']:+.2f}%")
    col2.metric("Positive Stocks", f"{market_stats['positive_stocks']}/{len(SYMBOLS)}")
    col3.metric("Total Volume", f"{market_stats['total_volume']/1e6:.1f}M")
    col4.metric("Market Status", "🟢 Open" if datetime.now().weekday() < 5 else "🔴 Closed")

    # Sector Performance Heatmap
    st.markdown('<p class="section-header">🎯 Daily Performance Heatmap</p>', unsafe_allow_html=True)
    market_data = []
    for symbol in SYMBOLS:
        hist, _ = get_stock_data(symbol, "5d")
        if hist is not None and len(hist) >= 2:
            change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            market_data.append({'Symbol': symbol, 'Company': COMPANY_NAMES.get(symbol, symbol), 'Change (%)': change})

    if market_data:
        market_df = pd.DataFrame(market_data)
        fig = px.bar(market_df, x='Symbol', y='Change (%)', color='Change (%)',
                     color_continuous_scale=px.colors.diverging.RdYlGn,
                     title="Daily Performance Overview", template='plotly_dark',
                     hover_data=['Company'])
        st.plotly_chart(fig, use_container_width=True)

    # News Sentiment Analysis Section
    st.markdown('<p class="section-header">📰 News Sentiment & Market Impact</p>', unsafe_allow_html=True)
    news_data = simulate_news_sentiment()

    col1, col2 = st.columns([2, 1])
    with col1:
        for news in news_data:
            sentiment_color = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#FFC107"}[news['sentiment']]
            trend_emoji = {"Bullish": "📈", "Bearish": "📉", "Neutral": "➡️"}[news['predicted_trend']]

            st.markdown(f"""
            <div class="news-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{news['symbol']}</strong> - {news['company']}<br>
                        <small>{news['headline']}</small>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: {sentiment_color}; font-weight: bold;">{news['sentiment']}</span><br>
                        <span>Impact: {news['impact_score']:.1f}/3.0</span><br>
                        <span>{trend_emoji} {news['predicted_trend']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Sentiment Distribution
        sentiment_counts = pd.DataFrame(news_data)['sentiment'].value_counts()
        fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                               title="News Sentiment Distribution", template='plotly_dark',
                               color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFC107'})
        st.plotly_chart(fig_sentiment, use_container_width=True)

def render_stock_analysis():
    """Renders the enhanced Stock Analysis page"""
    st.markdown('<p class="section-header">📈 Deep Dive Stock Analysis</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        selected_symbol = st.selectbox("Select a stock", SYMBOLS, key="analysis_stock")
    with c2:
        period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3, key="analysis_period")

    if selected_symbol:
        with st.spinner(f"Loading data for {selected_symbol}..."):
            hist, info = get_stock_data(selected_symbol, period)

        if hist is not None and not hist.empty:
            hist_with_indicators = add_technical_indicators(hist.copy())

            # Key Metrics Section
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change = current_price - prev_price

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
            col2.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
            if info and info.get('marketCap'):
                col3.metric("Market Cap", f"${info['marketCap']/1e9:.1f}B")
            if info and info.get('fiftyTwoWeekHigh'):
                col4.metric("52-Wk High", f"${info['fiftyTwoWeekHigh']:.2f}")

            # Price Chart
            fig_price = create_candlestick_chart(hist_with_indicators, f"{selected_symbol} - {COMPANY_NAMES.get(selected_symbol, selected_symbol)}")
            st.plotly_chart(fig_price, use_container_width=True)

            # Technical Indicators
            st.markdown('<p class="section-header">📊 Technical Indicators</p>', unsafe_allow_html=True)

            # RSI and MACD Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['RSI_14'],
                                             mode='lines', name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", template='plotly_dark')
                st.plotly_chart(fig_rsi, use_container_width=True)

            with col2:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['MACD'],
                                              mode='lines', name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['MACD_signal'],
                                              mode='lines', name='Signal', line=dict(color='red')))
                fig_macd.update_layout(title="MACD", yaxis_title="Value", template='plotly_dark')
                st.plotly_chart(fig_macd, use_container_width=True)

            # Additional Technical Indicators
            col1, col2 = st.columns(2)
            with col1:
                # Bollinger Bands
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['Close'],
                                            mode='lines', name='Close Price', line=dict(color='white')))
                fig_bb.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['BB_Upper'],
                                            mode='lines', name='BB Upper', line=dict(color='red', dash='dash')))
                fig_bb.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['BB_Lower'],
                                            mode='lines', name='BB Lower', line=dict(color='green', dash='dash')))
                fig_bb.update_layout(title="Bollinger Bands", yaxis_title="Price ($)", template='plotly_dark')
                st.plotly_chart(fig_bb, use_container_width=True)

            with col2:
                # Volume indicators
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=hist_with_indicators.index, y=hist_with_indicators['OBV'],
                                             mode='lines', name='OBV', line=dict(color='orange')))
                fig_vol.update_layout(title="On-Balance Volume (OBV)", yaxis_title="Volume", template='plotly_dark')
                st.plotly_chart(fig_vol, use_container_width=True)

            # BI Dashboard Section
            st.markdown('<p class="section-header">📊 Business Intelligence Dashboard</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="bi-placeholder">
                <h3>🚀 Advanced BI Dashboard Coming Soon!</h3>
                <p>This section will feature:</p>
                <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <li>Interactive financial ratios analysis</li>
                    <li>Peer comparison dashboards</li>
                    <li>Risk assessment metrics</li>
                    <li>Portfolio performance tracking</li>
                    <li>Custom KPI monitoring</li>
                </ul>
                <p><strong>Stay tuned for powerful business intelligence features!</strong></p>
            </div>
            """, unsafe_allow_html=True)

def render_price_prediction():
    """Renders the enhanced Price Prediction page"""
    st.markdown('<p class="section-header">🔮 AI-Powered Price Prediction</p>', unsafe_allow_html=True)

    selected_symbol = st.selectbox("Select a stock for prediction", SYMBOLS, key="prediction_stock")

    if selected_symbol:
        with st.spinner(f"Loading data and making prediction for {selected_symbol}..."):
            hist, _ = get_stock_data(selected_symbol, "1y")

            if hist is not None and not hist.empty:
                hist_with_indicators = add_technical_indicators(hist.copy())
                current_price = hist['Close'].iloc[-1]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("🎯 Enhanced Prediction Details")
                    st.markdown(f"**Analyzing:** {COMPANY_NAMES.get(selected_symbol, selected_symbol)}")

                    predicted_price, direction, confidence = predict_next_price(selected_symbol, hist_with_indicators)

                    if predicted_price is not None:
                        price_change = predicted_price - current_price
                        price_change_pct = (price_change / current_price) * 100

                        st.metric("Current Price", f"${current_price:.2f}",
                                  f"Last Update: {hist.index[-1].strftime('%Y-%m-%d')}")
                        st.metric("🔮 Predicted Price", f"${predicted_price:.2f}",
                                  f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

                        direction_color = "green" if direction == "Up" else "red" if direction == "Down" else "orange"
                        st.markdown(f"**Predicted Direction:** <span style='color: {direction_color}; font-weight: bold;'>{direction}</span>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**Direction Confidence:** {confidence:.1f}%")

                        # Enhanced progress bar with color coding
                        progress_color = "#4CAF50" if confidence >= 70 else "#FF9800" if confidence >= 50 else "#F44336"

                        # Custom progress bar with styling
                        st.markdown(f"""
                        <div style="width: 100%; background-color: #2E2E2E; border-radius: 10px; padding: 3px;">
                            <div style="width: {confidence:.1f}%; background-color: {progress_color};
                                        height: 25px; border-radius: 7px; transition: width 0.3s ease-in-out;
                                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                {confidence:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Prediction quality assessment
                        st.markdown("---")
                        if abs(price_change_pct) > 5: # Increased threshold for high volatility
                            quality_color = "#FF9800"
                            quality_text = "⚠️ **High Volatility Prediction** - Exercise caution"
                        elif confidence >= 75:
                            quality_color = "#4CAF50"
                            quality_text = "✅ **High Quality Prediction** - Strong model confidence"
                        elif confidence >= 60:
                            quality_color = "#FF9800"
                            quality_text = "⚡ **Moderate Quality Prediction** - Consider additional analysis"
                        else:
                            quality_color = "#F44336"
                            quality_text = "⚠️ **Low Confidence Prediction** - Use with caution"

                        st.markdown(f'<p style="color: {quality_color}; font-weight: bold;">{quality_text}</p>',
                                    unsafe_allow_html=True)

                        # Model insights
                        st.markdown("---")
                        st.markdown("**🧠 Model Architecture:**")
                        st.markdown("""
                        - **Conv1D layers** for pattern detection
                        - **Bidirectional GRU** for temporal modeling
                        - **Enhanced Attention** mechanism
                        - **Multi-task learning** (price + direction)
                        - **25+ technical indicators** as features
                        """)

                    else:
                        st.error("Could not generate a prediction. Check if model and scaler files exist.")

                with col2:
                    st.subheader("📈 Enhanced Forecast Visualization")
                    recent_data = hist_with_indicators.tail(60)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'],
                                             mode='lines+markers', name='Historical Price',
                                             line=dict(color='blue', width=2)))

                    if predicted_price is not None:
                        next_date = hist.index[-1] + timedelta(days=1)
                        fig.add_trace(go.Scatter(x=[hist.index[-1], next_date],
                                                 y=[current_price, predicted_price],
                                                 mode='lines+markers', name='AI Prediction',
                                                 line=dict(color='red', width=3, dash='dash'),
                                                 marker=dict(size=10, color='red')))

                        # Add confidence interval
                        confidence_interval = predicted_price * 0.02  # 2% confidence interval
                        fig.add_trace(go.Scatter(x=[next_date, next_date],
                                                 y=[predicted_price - confidence_interval,
                                                    predicted_price + confidence_interval],
                                                 mode='lines',
                                                 line=dict(color="rgba(255,0,0,0)"),
                                                 showlegend=False))

                        fig.add_trace(go.Scatter(
                            x=[next_date, next_date],
                            y=[predicted_price + confidence_interval, predicted_price - confidence_interval],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False
                        ))


                    fig.update_layout(title=f"{selected_symbol} - Enhanced AI Prediction",
                                      xaxis_title="Date", yaxis_title="Price ($)", height=450,
                                      template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)

            # Model Performance Metrics
            with st.expander("📊 Model Performance & Features"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🎯 Enhanced Features:**")
                    st.markdown("""
                    - **Price Data:** OHLCV + VWAP
                    - **Momentum:** RSI, ROC, Stochastic (%K, %D)
                    - **Trend:** SMA(20,50), MACD + Signal + Histogram
                    - **Volatility:** Bollinger Bands, ATR
                    - **Volume:** OBV, Chaikin Money Flow
                    - **Strength:** ADX (Average Directional Index)
                    - **Temporal:** Day/Month/Week features
                    """)

                with col2:
                    st.markdown("**🏗️ Architecture Improvements:**")
                    st.markdown("""
                    - **Conv1D Layers:** Pattern recognition in time series
                    - **Bidirectional GRU:** Forward + backward temporal analysis
                    - **Enhanced Attention:** Weighted feature importance
                    - **Multi-task Learning:** Price regression + direction classification
                    - **Regularization:** L2 + Dropout for better generalization
                    - **Advanced Optimization:** Adaptive learning rate
                    """)

                st.info(
                    f"""
                    **Enhanced Model Details for {selected_symbol}:**
                    - **Lookback Period:** {LOOK_BACK_PERIOD} days of historical data
                    - **Feature Count:** 25+ technical indicators and price features
                    - **Architecture:** CNN-BiGRU-Attention hybrid model
                    - **Training:** Multi-objective optimization with class balancing
                    - **Disclaimer:** This is an advanced ML model but not financial advice.
                    Always consult with qualified financial advisors before making investment decisions.
                    """
                )


# --- RAG AND AI CHATBOT SECTION (REVISED) ---

def setup_rag_chain(api_key):
    """Sets up the RAG chain for answering questions from the CSV file."""
    csv_file_path = 'data/newstock_data.csv' # Assuming the csv is in the 'data' folder
    if not os.path.exists(csv_file_path):
        st.error(f"Error: The data file '{csv_file_path}' was not found.")
        return None
    try:
        loader = CSVLoader(file_path=csv_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.6)
        prompt = ChatPromptTemplate.from_template("""
        You are an expert financial data analyst. Answer the user's question based *only* on the provided context from the stock data CSV file.
        Provide a clear, data-driven answer. If the context doesn't contain the answer, state that the information is not available in the provided data. Do not use outside knowledge.

        <context>
        {context}
        </context>

        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Failed to set up the RAG chain: {e}")
        return None

def setup_general_chat_chain(api_key):
    """Sets up the general purpose stock market chatbot."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are a helpful and knowledgeable AI assistant specializing in the US stock market. Your name is FinBot.
                - You can answer general questions about stocks, market trends, financial concepts, and news.
                - You are aware of the features of this dashboard, which include: detailed stock analysis with 25+ technical indicators, AI-powered price prediction, and news sentiment analysis. You can guide users on how to use these features.
                - When providing analysis, consider the potential impact of market sentiment. Use keywords like 'bullish', 'bearish', 'volatile', 'optimistic', 'pessimistic' where appropriate.
                - IMPORTANT: You are NOT a financial advisor. Do not give direct buy, sell, or hold recommendations. If asked for financial advice, you must decline and include the following disclaimer: 'Disclaimer: I am an AI assistant and not a qualified financial advisor. This information is for educational purposes only. Please consult with a professional for financial advice.'
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=ConversationBufferMemory(return_messages=True, memory_key="chat_history"),
            verbose=False
        )
        return chain
    except Exception as e:
        st.error(f"Failed to set up the General Chat chain: {e}")
        return None

def render_ai_chatbot():
    """Renders the AI Chatbot page with both RAG and General modes."""
    st.markdown('<p class="section-header">🤖 AI Financial Assistant</p>', unsafe_allow_html=True)

    # --- API KEY INPUT ---
    api_key_input = st.text_input(
        "Enter your Google API Key to activate the AI Assistant:",
        type="password",
        key="google_api_key_input",
        help="Get your key from Google AI Studio. Your key is not stored."
    )

    if api_key_input:
        st.session_state.google_api_key = api_key_input

    # --- CHATBOT INTERFACE (only if API key is provided) ---
    if "google_api_key" in st.session_state and st.session_state.google_api_key:
        
        # Initialize chains once after API key is provided
        if "rag_chain" not in st.session_state:
            with st.spinner("Initializing AI Data Analyst..."):
                st.session_state.rag_chain = setup_rag_chain(st.session_state.google_api_key)
        if "general_chain" not in st.session_state:
            with st.spinner("Initializing General Stock Assistant..."):
                st.session_state.general_chain = setup_general_chat_chain(st.session_state.google_api_key)
        
        st.markdown("---")
        # --- CHATBOT MODE SELECTOR ---
        mode = st.radio(
            "Select Assistant Mode:",
            ("AI Data Analyst (CSV-Powered)", "General Stock Assistant"),
            horizontal=True,
            help="Choose 'AI Data Analyst' to ask specific questions about the `newstock_data.csv` file. Choose 'General Stock Assistant' for broad questions about the market or this app."
        )

        # Initialize message histories
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [AIMessage(content="Hello! I am your AI Data Analyst. Ask me anything about the content of your `newstock_data.csv` file.")]
        if "general_messages_init" not in st.session_state:
             if st.session_state.general_chain:
                 st.session_state.general_chain.memory.chat_memory.add_ai_message("Hello! I am FinBot, your General Stock Assistant. How can I help you understand the market or this dashboard today?")
                 st.session_state.general_messages_init = True


        # --- RAG (CSV) CHATBOT INTERFACE ---
        if mode == "AI Data Analyst (CSV-Powered)":
            if st.session_state.rag_chain is None:
                st.error("Data Analyst Bot could not be initialized. Please check your API key or the `newstock_data.csv` file.")
                return

            st.info("You are chatting with the **AI Data Analyst**. It will only use information from the `newstock_data.csv` file to answer your questions.")
            messages = st.session_state.rag_messages
            chat_key = "rag_chat"

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    st.chat_message("user", avatar="👤").write(msg.content)
                else:
                    st.chat_message("assistant", avatar="🤖").write(msg.content)

            if prompt := st.chat_input("Ask a question about your newstock_data.csv file...", key=chat_key):
                messages.append(HumanMessage(content=prompt))
                st.chat_message("user", avatar="👤").write(prompt)

                with st.spinner("Analyzing your data..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"input": prompt})
                        ai_answer = response.get('answer', "I couldn't find an answer in the provided data.")
                        messages.append(AIMessage(content=ai_answer))
                        st.chat_message("assistant", avatar="🤖").write(ai_answer)
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {e}"
                        messages.append(AIMessage(content=error_msg))
                        st.chat_message("assistant", avatar="🤖").write(error_msg)
                st.rerun()

        # --- GENERAL STOCK ASSISTANT INTERFACE ---
        else:
            if st.session_state.general_chain is None:
                st.error("General Stock Assistant Bot could not be initialized. Please check your API Key.")
                return

            st.info("You are chatting with the **General Stock Assistant**. It can answer broad questions about the US stock market and this application.")
            messages = st.session_state.general_chain.memory.chat_memory.messages
            chat_key = "general_chat"

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    st.chat_message("user", avatar="👤").write(msg.content)
                else:
                    st.chat_message("assistant", avatar="🤖").write(msg.content)

            if prompt := st.chat_input("Ask a general stock market question...", key=chat_key):
                st.chat_message("user", avatar="👤").write(prompt)
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.general_chain.invoke({"input": prompt})
                        st.chat_message("assistant", avatar="🤖").write(response['response'])
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {e}"
                        st.chat_message("assistant", avatar="🤖").write(error_msg)
                st.rerun()
        
        st.markdown("---")
        # --- CLEAR CHAT BUTTON ---
        if st.button("🗑️ Clear Conversation"):
            if mode == "AI Data Analyst (CSV-Powered)":
                st.session_state.rag_messages = [AIMessage(content="Hello! I am your AI Data Analyst. Ask me anything about the content of your `newstock_data.csv` file.")]
            else:
                # Re-initialize the chain to clear memory
                if "google_api_key" in st.session_state:
                     st.session_state.general_chain = setup_general_chat_chain(st.session_state.google_api_key)
                     st.session_state.general_messages_init = False # Re-trigger welcome message
            st.rerun()

    else:
        st.warning("Please enter your Google API Key above to activate the AI Financial Assistant.")

# ----------------- MODEL INFORMATION PAGE -----------------
def render_model_information():
    """Renders a page detailing information about the models used."""
    st.markdown('<p class="section-header">🧠 AI Model Information</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
                 border: 1px solid #00A0B0; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem;">
        <h3 style="color: #00A0B0; margin-top: 0;">Overview of AI Models in this Dashboard</h3>
        <p>This dashboard leverages multiple sophisticated AI models to provide comprehensive stock market analysis, prediction, and news sentiment understanding.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Stock Price Prediction Model ---
    st.markdown('<p class="section-header">🔮 Stock Price Prediction Model</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Hybrid CNN-BiGRU-Attention Model</h4>
        <p>This model is designed for time series forecasting, specifically predicting stock price movements and direction. It combines the strengths of several neural network architectures:</p>
        <ul>
            <li><strong>Convolutional Neural Networks (CNN):</strong> Effective at identifying local patterns and features in time-series data.</li>
            <li><strong>Bidirectional Gated Recurrent Units (BiGRU):</strong> Captures temporal dependencies in both forward and backward directions, understanding context from past and future data points in a sequence.</li>
            <li><strong>Attention Mechanism:</strong> Allows the model to dynamically weigh the importance of different time steps in the input sequence, focusing on the most relevant historical data for prediction.</li>
        </ul>
        <p><strong>Key Features & Inputs:</strong></p>
        <ul>
            <li><strong>Lookback Period:</strong> Utilizes 100 days of historical data for each prediction.</li>
            <li><strong>Extensive Features:</strong> Incorporates 25+ technical indicators including:
                <ul>
                    <li>Price-based (OHLCV, VWAP)</li>
                    <li>Momentum (RSI, ROC, Stochastic)</li>
                    <li>Trend (SMA, MACD)</li>
                    <li>Volatility (Bollinger Bands, ATR)</li>
                    <li>Volume-based (OBV, CMF)</li>
                    <li>Strength (ADX)</li>
                    <li>Temporal features (Day of Week, Month of Year, Week of Year)</li>
                </ul>
            </li>
            <li><strong>Multi-task Learning:</strong> The model is trained to simultaneously predict the next day's logarithmic return and the direction of the price movement (Up, Down, Sideways), leading to more robust predictions.</li>
            <li><strong>Regularization:</strong> Employs L2 regularization and Dropout layers to prevent overfitting and improve generalization.</li>
            <li><strong>Optimization:</strong> Uses advanced optimizers with adaptive learning rates.</li>
        </ul>
        <p><strong>Output:</strong> Predicts the next trading day's closing price and the likely direction (Up, Down, or Sideways) with a confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- News Sentiment Analysis Model ---
    st.markdown('<p class="section-header">📰 News Sentiment Analysis Model</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>FinBERT for Sentiment + XGBoost for Impact</h4>
        <p>This component analyzes financial news headlines to determine their sentiment and predict their potential impact on stock prices:</p>
        <ul>
            <li><strong>FinBERT (Financial Bidirectional Encoder Representations from Transformers):</strong> A state-of-the-art Natural Language Processing (NLP) model pre-trained on a large corpus of financial text. It is fine-tuned for financial sentiment classification (Positive, Negative, Neutral).</li>
            <li><strong>XGBoost (Extreme Gradient Boosting):</strong> A powerful gradient boosting machine learning algorithm used to predict the quantitative price impact (percentage change) based on the sentiment output from FinBERT.</li>
        </ul>
        <p><strong>Process:</strong></p>
        <ol>
            <li>**Headline Cleaning:** Raw headlines are cleaned by removing URLs, special characters, and converting to lowercase.</li>
            <li>**Sentiment Prediction:** The cleaned headline is fed into the FinBERT model, which outputs a sentiment label (Positive, Negative, Neutral) and a confidence score for that sentiment.</li>
            <li>**Price Impact Prediction:** The sentiment label is then used by the XGBoost model to estimate the percentage change in the stock price due to the news. This impact is further refined by weighting it with the FinBERT's confidence and applying realistic market bounds.</li>
        </ol>
        <p><strong>Output:</strong> Sentiment (Positive/Negative/Neutral), Confidence in Sentiment, Predicted Price Impact (%), and Predicted Price for the stock.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- AI Chatbot Models ---
    st.markdown('<p class="section-header">🤖 AI Chatbot Models</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Google Gemini 1.5 Flash</h4>
        <p>The AI Chatbot feature utilizes Google's powerful Gemini 1.5 Flash model, a highly efficient and capable large language model (LLM).</p>
        <ul>
            <li><strong>General Stock Assistant:</strong> This chatbot is directly powered by Gemini 1.5 Flash, enabling it to answer a wide range of general questions about the US stock market, financial concepts, and the features available within this dashboard. It's configured with a system prompt to act as a helpful financial assistant while adhering to strict disclaimers against providing financial advice.</li>
            <li><strong>AI Data Analyst (CSV-Powered):</strong> This specialized chatbot uses Gemini 1.5 Flash in conjunction with a Retrieval Augmented Generation (RAG) framework.
                <ul>
                    <li>**RAG with FAISS and Google Generative AI Embeddings:** It first converts your `newstock_data.csv` content into numerical embeddings. These embeddings are stored in a FAISS vector database.</li>
                    <li>When you ask a question, the relevant chunks of your CSV data are retrieved and provided to Gemini 1.5 Flash as context. This ensures that the AI answers *only* from your specific data, functioning as a "data analyst" for your CSV.</li>
                </ul>
            </li>
        </ul>
        <p><strong>Benefits:</strong> Provides accurate, context-aware, and safe responses, enhancing user interaction and data exploration capabilities.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Disclaimer: All AI predictions and analyses are for informational and educational purposes only and do not constitute financial advice. Investment decisions should always be made with careful consideration and consultation with a qualified financial advisor.")


# ----------------- MAIN APP LOGIC -----------------
def main():
    # --- NEW: Call the download function at the start of the app ---
    download_and_unzip_files()
    
    st.markdown('<h1 class="main-header">📈 Stock Market Price Prediction and News Sentiment Analysis</h1>', unsafe_allow_html=True)

    # --- TOP NAVIGATION BAR ---
    page = option_menu(
        menu_title=None,
        options=["Market Overview", "Stock Analysis", "Price Prediction", "News Sentiment", "AI Chatbot", "Model Information"],
        icons=["house-door-fill", "clipboard-data-fill", "cpu-fill", "newspaper", "robot", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#121212"},
            "icon": {"color": "#00A0B0", "font-size": "22px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#2E2E2E",
                "font-weight": "bold",
            },
            "nav-link-selected": {"background-color": "#007BFF"},
        }
    )

    # --- PAGE ROUTING ---
    if page == "Market Overview":
        render_market_overview()
    elif page == "Stock Analysis":
        render_stock_analysis()
    elif page == "Price Prediction":
        render_price_prediction()
    elif page == "News Sentiment":
        render_news_sentiment_analysis()
    elif page == "AI Chatbot":
        render_ai_chatbot()
    elif page == "Model Information":
        render_model_information()

if __name__ == "__main__":
    main()
