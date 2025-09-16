import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, GRU, Bidirectional, Dense, Dropout, Layer,
                                     Conv1D, BatchNormalization, LeakyReLU)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# --- Configuration ---
SYMBOLS = [
    'AAPL', 'NVDA', 'TSLA', 'AMZN', 'MSFT',
    'META', 'GOOGL', 'AMD', 'NFLX', 'JPM'
]
DATA_FILE = 'newstock_data.csv'

# --- Model Hyperparameters ---
LOOK_BACK_PERIOD = 100
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 250
PATIENCE = 40
LEARNING_RATE = 0.001
NEUTRAL_THRESHOLD = 0.005

# --- Directory Setup ---
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('scalers'): os.makedirs('scalers')
if not os.path.exists('plots'): os.makedirs('plots')


# --- Custom Attention Layer (Unchanged) ---
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def add_technical_indicators(df):
    """Adds a comprehensive set of technical indicators to the dataframe."""
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    low_14, high_14 = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(3).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    exp1, exp2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/14).mean()) / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX_14'] = dx.ewm(alpha=1/14).mean()
    sma_20_bb, std_dev_20 = df['Close'].rolling(window=20).mean(), df['Close'].rolling(window=20).std()
    df['BB_Upper'], df['BB_Lower'] = sma_20_bb + (std_dev_20 * 2), sma_20_bb - (std_dev_20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma_20_bb
    df['ATR'] = atr
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfv.fillna(0)
    mfv *= df['Volume']
    df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month_of_year'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def create_direction_target(df, threshold):
    future_close = df['Close'].shift(-1)
    log_return = np.log(future_close / df['Close'])
    conditions = [
        log_return < -threshold,
        (log_return >= -threshold) & (log_return <= threshold),
        log_return > threshold
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=1)

def load_and_prepare_data(filepath, symbol):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df_symbol = df[df['Symbol'] == symbol].copy()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_symbol.dropna(subset=required_cols, inplace=True)
        df_symbol.sort_values('Date', inplace=True)
        df_symbol.reset_index(drop=True, inplace=True)
        df_symbol = add_technical_indicators(df_symbol)
        df_symbol['Log_Return'] = np.log(df_symbol['Close'] / df_symbol['Close'].shift(1))
        df_symbol['Direction_Target'] = create_direction_target(df_symbol, NEUTRAL_THRESHOLD)
        df_symbol.dropna(inplace=True)
        feature_cols = [
            'Close', 'Open', 'High', 'Low', 'Volume', 'VWAP', 'CMF',
            'ROC', 'RSI_14', '%K', '%D', 'SMA_20', 'SMA_50', 'MACD',
            'MACD_signal', 'MACD_hist', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'ATR', 'OBV', 'ADX_14', 'day_of_week', 'month_of_year', 'week_of_year'
        ]
        return df_symbol, feature_cols
    except Exception as e:
        print(f"An error occurred while loading data for {symbol}: {e}")
        return None, None

def create_sequences(features, log_return_target, direction_target, look_back):
    X, y_log_return, y_direction = [], [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back), :])
        y_log_return.append(log_return_target[i + look_back])
        y_direction.append(direction_target[i + look_back])
    return np.array(X), np.array(y_log_return), np.array(y_direction)

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    L2_REG = 0.0055
    x = Conv1D(filters=64, kernel_size=5, padding='causal', kernel_regularizer=l2(L2_REG))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(L2_REG)))(x)
    x = Dropout(0.45)(x)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(L2_REG)))(x)
    x = Dropout(0.45)(x)
    attention_output = Attention()(x)
    shared_dense = Dense(64, kernel_regularizer=l2(L2_REG))(attention_output)
    shared_dense = LeakyReLU(negative_slope=0.1)(shared_dense)
    price_branch = Dense(32, kernel_regularizer=l2(L2_REG))(shared_dense)
    price_branch = LeakyReLU(negative_slope=0.1)(price_branch)
    price_output = Dense(1, name='price')(price_branch)
    direction_branch = Dense(64, kernel_regularizer=l2(L2_REG))(shared_dense)
    direction_branch = BatchNormalization()(direction_branch)
    direction_branch = LeakyReLU(negative_slope=0.1)(direction_branch)
    direction_branch = Dropout(0.3)(direction_branch)
    direction_branch = Dense(32, kernel_regularizer=l2(L2_REG))(direction_branch)
    direction_branch = LeakyReLU(negative_slope=0.1)(direction_branch)
    direction_output = Dense(3, activation='softmax', name='direction')(direction_branch)
    
    # --- FIXED: Define model outputs as a dictionary to match compile/fit ---
    model = Model(inputs=inputs, outputs={'price': price_output, 'direction': direction_output})
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss={'price': 'huber', 'direction': 'categorical_crossentropy'},
                  loss_weights={'price': 0.10, 'direction': 0.90},
                  metrics={'price': 'mean_absolute_error', 'direction': 'accuracy'})
    return model

def predict_next_day_price(symbol, full_df, feature_cols):
    """
    Loads the trained model and scaler to predict the next trading day's closing price.
    """
    print(f"\n--- Predicting Next Day's Price for {symbol} ---")
    try:
        model_path = os.path.join('models', f'{symbol}_best_model.keras')
        scaler_path = os.path.join('scalers', f'{symbol}_feature_scaler.pkl')
        custom_objects = {'Attention': Attention}
        model = load_model(model_path, custom_objects=custom_objects)
        scaler = joblib.load(scaler_path)
        last_sequence_df = full_df.iloc[-LOOK_BACK_PERIOD:]
        last_sequence_features = last_sequence_df[feature_cols].values
        last_actual_price = last_sequence_df['Close'].iloc[-1]
        scaled_features = scaler.transform(last_sequence_features)
        input_data = np.reshape(scaled_features, (1, LOOK_BACK_PERIOD, len(feature_cols)))
        prediction = model.predict(input_data)
        
        # When model outputs are a dictionary, predictions are also a dictionary
        predicted_log_return = prediction['price'][0][0]
        predicted_direction_probs = prediction['direction'][0]
        
        predicted_price = last_actual_price * np.exp(predicted_log_return)
        direction_map = {0: 'Down', 1: 'Sideways', 2: 'Up'}
        predicted_direction_index = np.argmax(predicted_direction_probs)
        predicted_direction = direction_map[predicted_direction_index]
        print(f"Last available closing price ({full_df['Date'].iloc[-1].date()}): ${last_actual_price:.2f}")
        print(f"Predicted Next Day's Closing Price for {symbol}: ${predicted_price:.2f}")
        print(f"Predicted Direction: {predicted_direction} (Confidence: {predicted_direction_probs[predicted_direction_index]:.2%})")
    except FileNotFoundError:
        print(f"Model or scaler for {symbol} not found. Cannot make a prediction.")
    except Exception as e:
        print(f"An error occurred during next-day prediction for {symbol}: {e}")

def main():
    report_filename = f"training_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}\nProcessing Symbol: {symbol}\n{'='*60}\n")

        full_df, feature_cols = load_and_prepare_data(DATA_FILE, symbol)
        if full_df is None or len(full_df) < LOOK_BACK_PERIOD * 3:
            print(f"Insufficient data for {symbol} after processing. Skipping.")
            continue
        
        train_end = int(len(full_df) * TRAIN_RATIO)
        val_end = int(len(full_df) * (TRAIN_RATIO + VALIDATION_RATIO))
        train_df = full_df.iloc[:train_end]
        val_df = full_df.iloc[train_end:val_end]
        test_df = full_df.iloc[val_end:]

        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_train_features = feature_scaler.fit_transform(train_df[feature_cols])
        scaled_val_features = feature_scaler.transform(val_df[feature_cols])
        scaled_test_features = feature_scaler.transform(test_df[feature_cols])
        
        X_train, y_train_log_return, y_train_dir = create_sequences(scaled_train_features, train_df['Log_Return'].values, train_df['Direction_Target'].values, LOOK_BACK_PERIOD)
        X_val, y_val_log_return, y_val_dir = create_sequences(scaled_val_features, val_df['Log_Return'].values, val_df['Direction_Target'].values, LOOK_BACK_PERIOD)
        X_test, y_test_log_return, y_test_dir = create_sequences(scaled_test_features, test_df['Log_Return'].values, test_df['Direction_Target'].values, LOOK_BACK_PERIOD)
        
        y_train_dir_cat = to_categorical(y_train_dir, num_classes=3)
        y_val_dir_cat = to_categorical(y_val_dir, num_classes=3)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print(f"Not enough data to create sequences for {symbol} splits. Skipping.")
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]))
        model_path = os.path.join('models', f'{symbol}_best_model.keras') 
        
        callbacks = [
            ModelCheckpoint(model_path, monitor='val_direction_accuracy', save_best_only=True, mode='max', save_weights_only=False, verbose=1),
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-7, verbose=1)
        ]
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_dir), y=y_train_dir)
        class_weights_dict = dict(enumerate(class_weights))
        print(f"Using Class Weights for Directional Classification: {class_weights_dict}")
        
        direction_sample_weights = np.array([class_weights_dict[c] for c in y_train_dir])
        price_sample_weights = np.ones(X_train.shape[0])
        
        model.fit(X_train, {'price': y_train_log_return, 'direction': y_train_dir_cat},
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_val, {'price': y_val_log_return, 'direction': y_val_dir_cat}),
                  callbacks=callbacks,
                  sample_weight={'price': price_sample_weights, 'direction': direction_sample_weights},
                  verbose=1)

        print(f"\n--- Evaluating Best Model for {symbol} on Test Set ---")
        predictions = model.predict(X_test)
        
        # Because the model now outputs a dictionary, the predictions will also be a dictionary
        predicted_log_returns = predictions['price'].flatten()
        predicted_dir_probs = predictions['direction']
        
        last_day_prices = test_df['Close'].values[LOOK_BACK_PERIOD-1:-1]
        predicted_prices = last_day_prices * np.exp(predicted_log_returns)
        actual_prices = test_df['Close'].values[LOOK_BACK_PERIOD:]
        
        predicted_direction = np.argmax(predicted_dir_probs, axis=1)
        actual_direction = y_test_dir
        
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        dir_accuracy = accuracy_score(actual_direction, predicted_direction) * 100
        precision = precision_score(actual_direction, predicted_direction, average='macro', zero_division=0) * 100
        recall = recall_score(actual_direction, predicted_direction, average='macro', zero_division=0) * 100
        f1 = f1_score(actual_direction, predicted_direction, average='macro', zero_division=0) * 100

        print(f"Root Mean Squared Error (RMSE) on Price: {rmse:.4f}")
        print(f"Directional Accuracy: {dir_accuracy:.2f}%")
        print(f"Precision (Macro): {precision:.2f}%")
        print(f"Recall (Macro): {recall:.2f}%")
        print(f"F1-Score (Macro): {f1:.2f}%")

        plt.figure(figsize=(15, 7))
        plt.plot(test_df['Date'].iloc[LOOK_BACK_PERIOD:], actual_prices, color='royalblue', label='Actual Stock Price')
        plt.plot(test_df['Date'].iloc[LOOK_BACK_PERIOD:], predicted_prices, color='crimson', label='Predicted Stock Price', alpha=0.8)
        plt.title(f'{symbol} Stock Price Prediction (Test Set)', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Stock Price (USD)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path_prices = os.path.join('plots', f'{symbol}_prediction_vs_actual.png')
        plt.savefig(plot_path_prices)
        plt.close()
        print(f"Saved price prediction plot to: {plot_path_prices}")

        results.append({
            'symbol': symbol, 'rmse': rmse, 'dir_accuracy': dir_accuracy,
            'precision': precision, 'recall': recall, 'f1': f1
        })
        joblib.dump(feature_scaler, os.path.join('scalers', f'{symbol}_feature_scaler.pkl'))
        
        predict_next_day_price(symbol, full_df, feature_cols)

    print(f"\n{'='*60}\n--- Final Training Report ---\n{'='*60}\n")
    header = f"{'Symbol':<10} | {'RMSE':<12} | {'Accuracy (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<12} | {'F1-Score (%)'}"
    separator = "-" * len(header)
    
    with open(report_filename, 'w') as f:
        f.write(f"Training Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        print(header)
        print(separator)
        
        for result in results:
            line = (f"{result['symbol']:<10} | {result['rmse']:<12.4f} | {result['dir_accuracy']:<15.2f} | "
                    f"{result['precision']:<15.2f} | {result['recall']:<12.2f} | {result['f1']:.2f}")
            f.write(line + "\n")
            print(line)
            
    print(f"\nFull report saved to: {report_filename}")

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Running on CPU.")
    main()