# AI Stock Prediction and Sentiment Analysis Dashboard

This project is a comprehensive stock market analysis tool that uses AI to predict future stock prices, analyze news sentiment, and provide a detailed dashboard for market overview.

## Features

- **AI Price Prediction:** Uses a CNN-BiGRU-Attention model to forecast the next day's stock price and market direction.
- **News Sentiment Analysis:** Employs a fine-tuned FinBERT model to determine the sentiment of financial news and predict its impact on stock prices.
- **Interactive Dashboard:** A Streamlit application that visualizes market data, technical indicators, and AI predictions.
- **AI Chatbot:** A financial assistant powered by Google's Gemini to answer questions about market data.

## Project Structure

```
your-project-name/
│
├── data/                 # Folder for datasets
├── models/               # Saved price prediction models (.keras)
├── scalers/              # Saved data scalers (.pkl)
├── sentiment_models/     # Saved FinBERT and impact models
│
├── .gitignore            # Files to be ignored by Git
├── requirements.txt      # Python dependencies
├── README.md             # Project description
├── train_price_model.py  # Script to train the price prediction models
├── train_sentiment_model.py # Script to train the sentiment analysis models
└── app.py                # Main Streamlit application file
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    cd YOUR_REPOSITORY
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Train the AI Models:**
    *First, run the price prediction model training script:*
    ```bash
    python train_price_model.py
    ```
    *Next, run the sentiment analysis model training script:*
    ```bash
    python train_sentiment_model.py
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```