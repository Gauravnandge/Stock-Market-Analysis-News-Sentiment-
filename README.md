# 📈Stock Market Price Prediction and News Sentiment Analysis 

This is a comprehensive, interactive web application built with Streamlit for stock market analysis. It leverages multiple AI and Machine Learning models to provide deep insights, including real-time data visualization, AI-powered price prediction, news sentiment analysis with price impact, and an integrated AI financial assistant.



---

## ✨ Key Features

* **🌍 Market Overview:** Get a real-time snapshot of the market with major indices (S&P 500, Dow Jones, NASDAQ), top daily gainers and losers, and a performance heatmap of key stocks.
* **📊 Deep Dive Stock Analysis:** Select any stock to view detailed interactive charts, including candlestick patterns, volume, and over 25 technical indicators like RSI, MACD, Bollinger Bands, SMA, and more.
* **🔮 Price Prediction:** Utilizes a sophisticated hybrid **CNN-BiGRU-Attention** model to forecast the next day's stock price and market direction (Up, Down, Sideways) with a confidence score.
* **📰 News Sentiment Analysis:** Analyzes any financial news headline using a fine-tuned **FinBERT** model to determine the sentiment (Positive, Negative, Neutral) and predicts the potential price impact using an XGBoost model.
* **🤖 Financial Assistant:** A built-in chatbot powered by **Google's Gemini 1.5 Flash** with two modes:
    * **General Assistant:** Answers broad questions about market trends, financial concepts, and how to use the dashboard.
    * **Data Analyst (RAG):** Answers specific questions about your own data by querying an uploaded `newstock_data.csv` file using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🛠️ Tech Stack & Architecture

* **Frontend:** Streamlit
* **Data Source:** `yfinance` for real-time stock data.
* **Data Science & ML:** Pandas, NumPy, Scikit-learn
* **Deep Learning (Price Prediction):** TensorFlow / Keras
* **NLP (Sentiment Analysis):** PyTorch, Hugging Face Transformers (FinBERT)
* **AI Chatbot & RAG:** LangChain, Google Generative AI (Gemini 1.5 Flash), FAISS (for vector storage)
* **Plotting:** Plotly
* **Deployment Helper:** `gdown` for automatic model downloading from Google Drive.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9 or higher
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Gauravnandge/Stock-Market-Analysis-News-Sentiment-.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the content below and run the installation command.

    **`requirements.txt`:**
    ```
    streamlit
    yfinance
    pandas
    numpy
    plotly
    tensorflow
    scikit-learn
    joblib
    streamlit-option-menu
    torch
    transformers
    langchain-google-genai
    langchain
    langchain-community
    faiss-cpu
    gdown
    ```

    **Installation command:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Automatic Model Download:**
    The first time you run the application, it will automatically download the necessary model files (`~500-700MB`) from a shared Google Drive link. This is a one-time setup process. Please be patient as it may take a few minutes.

2.  **Run the Streamlit app:**
    *(Assuming your main script is named `app.py`)*
    ```bash
    streamlit run app.py
    ```

3.  **Use the AI Chatbot:**
    To activate the AI Financial Assistant, you will need a **Google API Key**.
    * You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Navigate to the "AI Chatbot" page in the app and paste your key into the input field. The key is used for the session and is not stored.

---

## 📁 Project Structure

The application automatically creates the following directory structure upon first run by downloading the necessary assets.
 ```
.
├── sentiment_models/
│   ├── finbert_stock_sentiment_model/
│   │   └── ... (FinBERT model files)
│   ├── impact_prediction_model.pkl
│   └── label_encoder.pkl
├── models/
│   ├── AAPL_best_model.keras
│   └── ... (Other prediction models)
├── scalers/
│   ├── AAPL_feature_scaler.pkl
│   └── ... (Other feature scalers)
├── data/
│   └── newstock_data.csv (For the RAG chatbot)
├── plots/
│   └── ... (Any saved plots)
├── app.py
├── requirements.txt
└── README.md
 ```
---

## ⚖️ Disclaimer

This project is for educational and informational purposes only. The predictions and analyses provided by the AI models **do not constitute financial advice**. Trading and investing in financial markets involve substantial risk. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.
