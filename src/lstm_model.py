# src/lstm_model.py - LSTM for SPY sequence prediction
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load features (focus on close_spy for sequence pred)
df = pd.read_csv('data/features.csv', parse_dates=True, index_col='timestamp')

# Guard: Check data
if len(df) == 0:
    print("Error: features.csv empty. Rerun feature_eng.py.")
    exit()

# Scale closes (0-1 for stable training)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df['close_spy'].values.reshape(-1,1))

# Reshape to sequences (lookback 60 days—captures trends)
lookback = 60  # Tweak if needed (e.g., 30 for shorter memory)
X, y = [], []
for i in range(lookback, len(scaled)):
    X.append(scaled[i-lookback:i, 0])  # Sequence of past 60 closes
    y.append(scaled[i, 0])  # Next close
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # 3D: (samples, timesteps, feats)

# Train/test split (time-series: 80/20, no shuffle)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model (2 layers for depth, Dense for output)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))  # Layer 1: Returns seq for next layer
model.add(LSTM(50))  # Layer 2: Captures deeper patterns
model.add(Dense(1))  # Output: Next scaled close
model.compile(optimizer='adam', loss='mean_squared_error')  # Adam good for time-series

# Train (20 epochs—balance learning/overfit)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)  # Val split monitors overfit

# Predict/evaluate MSE (inverse scale for real values)
preds_scaled = model.predict(X_test)
preds = scaler.inverse_transform(preds_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1,1))
mse = mean_squared_error(y_true, preds)
print(f"LSTM MSE: {mse:.4f}")  # Compare to XGB ~3.6192

# Save model
model.save('models/lstm_model.h5')

# Compare to XGB MSE (from ml_models run)
xgb_mse = 3.6192  # Replace with your latest
print(f"LSTM vs XGB MSE: LSTM better by {xgb_mse - mse:.4f}" if mse < xgb_mse else "XGB better")