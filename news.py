import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import re
import joblib
import os

def train_finbert_sentiment_and_impact_model():
    """
    Main function to load data, train sentiment and impact models, and save the artifacts.
    """
    # 1. Load and Preprocess Data
    # ==================================
    try:
        df = pd.read_csv('aligned_news_data.csv')
    except FileNotFoundError:
        print("Error: 'aligned_news_data.csv' not found. Please ensure the file is in the correct directory.")
        return

    print("Step 1: Preprocessing data...")
    symbols = ['NVDA', 'TSLA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOGL', 'AMD', 'NFLX', 'JPM']
    df = df[df['Symbol'].isin(symbols)]

    def clean_headline(headline):
        if not isinstance(headline, str):
            return ""
        headline = re.sub(r'\[.*?\]', '', headline)  # Remove text in square brackets
        headline = re.sub(r'http\S+', '', headline)  # Remove URLs
        headline = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', headline)  # Keep some punctuation
        headline = headline.lower()  # Convert to lowercase
        return headline.strip()

    df['cleaned_headline'] = df['Headline'].apply(clean_headline)
    df.dropna(subset=['cleaned_headline'], inplace=True)
    df = df[df['cleaned_headline'] != '']

    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['finbert_sentiment'])

    # 2. Fetch Historical Stock Data and Calculate Impact Percentage
    # ==============================================================
    print("Step 2: Fetching historical stock data and calculating impact...")
    all_stock_data = []
    for symbol in tqdm(symbols, desc="Fetching stock data"):
        try:
            stock = yf.Ticker(symbol)
            # Fetch data from 2000 onwards to cover the dataset range
            hist = stock.history(start="2000-01-01")
            if not hist.empty:
                hist['Symbol'] = symbol
                all_stock_data.append(hist)
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {e}")

    if not all_stock_data:
        print("Could not fetch any stock data. Exiting.")
        return

    stock_df = pd.concat(all_stock_data)
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = stock_df['Date'].dt.strftime('%m-%d-%Y')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m-%d-%Y').dt.strftime('%m-%d-%Y')
    df.dropna(subset=['Date'], inplace=True)
    
    merged_df = pd.merge(df, stock_df[['Date', 'Symbol', 'Close']], on=['Date', 'Symbol'], how='left')
    merged_df.sort_values(by=['Symbol', 'Date'], inplace=True)

    merged_df['next_day_close'] = merged_df.groupby('Symbol')['Close'].shift(-1)
    merged_df.dropna(subset=['Close', 'next_day_close'], inplace=True)
    merged_df['impact_percentage'] = ((merged_df['next_day_close'] - merged_df['Close']) / merged_df['Close']) * 100
    
    # 3. FinBERT Sentiment Model Training
    # ===================================
    print("\nStep 3: Starting FinBERT sentiment model training...")
    X_train, X_val, y_train, y_val = train_test_split(
        merged_df['cleaned_headline'], merged_df['sentiment_encoded'],
        test_size=0.2, random_state=42, stratify=merged_df['sentiment_encoded']
    )

    class NewsSentimentDataset(Dataset):
        def __init__(self, headlines, labels, tokenizer, max_len):
            self.headlines = headlines
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.headlines)

        def __getitem__(self, item):
            headline = str(self.headlines[item])
            label = self.labels[item]
            encoding = self.tokenizer.encode_plus(
                headline, add_special_tokens=True, max_length=self.max_len,
                return_token_type_ids=False, padding='max_length',
                return_attention_mask=True, return_tensors='pt', truncation=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def create_data_loader(headlines, labels, tokenizer, max_len, batch_size):
        ds = NewsSentimentDataset(headlines=headlines, labels=labels, tokenizer=tokenizer, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, num_workers=0) # num_workers=0 for Windows compatibility

    PRE_TRAINED_MODEL_NAME = 'ProsusAI/finbert'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, 
        num_labels=len(label_encoder.classes_),
        use_safetensors=True
    )

    BATCH_SIZE = 16
    MAX_LEN = 160 # Increased max length for potentially longer headlines
    train_data_loader = create_data_loader(X_train.to_numpy(), y_train.to_numpy(), tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(X_val.to_numpy(), y_val.to_numpy(), tokenizer, MAX_LEN, BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    EPOCHS = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for d in progress_bar:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'training_loss': f'{loss.item():.3f}'})

    # 4. Sentiment Model Evaluation
    # =============================
    print("\nStep 4: Evaluating sentiment model...")
    model.eval()
    predictions, real_values = [], []
    with torch.no_grad():
        for d in tqdm(val_data_loader, desc="Evaluating"):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    print("\nSentiment Model Evaluation Report:")
    print(f"Accuracy: {accuracy_score(real_values, predictions):.4f}")
    print(classification_report(real_values, predictions, target_names=label_encoder.classes_))

    # 5. Impact Percentage Prediction Model
    # =====================================
    print("Step 5: Training impact prediction model...")
    def get_all_predictions(headlines, labels):
        all_data_loader = create_data_loader(headlines, labels, tokenizer, MAX_LEN, BATCH_SIZE)
        model.eval()
        predictions = []
        with torch.no_grad():
            for d in tqdm(all_data_loader, desc="Getting all sentiment predictions"):
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds)
        return torch.stack(predictions).cpu()

    merged_df['predicted_sentiment'] = get_all_predictions(merged_df['cleaned_headline'].to_numpy(), merged_df['sentiment_encoded'].to_numpy())
    X_impact = merged_df[['predicted_sentiment']]
    y_impact = merged_df['impact_percentage']

    X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_impact, y_impact, test_size=0.2, random_state=42)

    impact_model = LinearRegression()
    impact_model.fit(X_train_imp, y_train_imp)
    y_pred_imp = impact_model.predict(X_test_imp)

    print("\nImpact Percentage Model Evaluation:")
    mse = mean_squared_error(y_test_imp, y_pred_imp)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

    # 6. Save Models and Tokenizer
    # ============================
    print("\nStep 6: Saving models and artifacts...")
    output_dir = './finbert_stock_sentiment_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(impact_model, 'impact_prediction_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print(f"Models saved successfully to '{output_dir}' and local files.")

    # Display a sample of predictions
    print("\n--- Final Sample Predictions ---")
    sample_df = merged_df.sample(min(10, len(merged_df)))
    sample_df['predicted_sentiment_label'] = label_encoder.inverse_transform(sample_df['predicted_sentiment'])
    sample_df['predicted_impact_percentage'] = impact_model.predict(sample_df[['predicted_sentiment']])
    print(sample_df[['Date', 'Symbol', 'cleaned_headline', 'predicted_sentiment_label', 'impact_percentage', 'predicted_impact_percentage']].to_string())

if __name__ == '__main__':
    train_finbert_sentiment_and_impact_model()

