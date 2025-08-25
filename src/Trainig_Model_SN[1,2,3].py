import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

def plot_residuals(sn, df, model, features, target='thrust', n_points=300):
    """
    Plot residuals (Actual - Predicted) for a specific serial number (SN).

    This function calculates and visualizes the residuals between the true and predicted values
    for a subset of the dataset corresponding to a given serial number.  
    If the number of available data points is less than `n_points`, the function displays a warning
    and does not produce a plot.

    Args:
        sn (int | str): Serial number identifying the subset of data to be analyzed.
        df (pd.DataFrame): Input dataframe containing the dataset. Must include columns listed in `features` and `target`.
        model (object): Trained model with a `.predict()` method.
        features (list[str]): List of column names used as model input features.
        target (str, optional): Name of the target column (ground truth). Defaults to `'thrust'`.
        n_points (int, optional): Maximum number of points to display in the plot. Defaults to 300.

    Returns:
        None: Displays a residual plot and does not return any value.

    Raises:
        ValueError: If required columns are missing from the dataframe.

    Example:
        >>> plot_residuals(
        ...     sn=101,
        ...     df=dataframe,
        ...     model=trained_model,
        ...     features=["pressure", "temperature"],
        ...     target="thrust",
        ...     n_points=200
        ... )
    """
    # Filter data for the given SN and remove rows with missing values
    df_sn = df[df['sn'] == sn].dropna(subset=features + [target])

    # Check if there are enough data points
    if len(df_sn) < n_points:
        print(f"⚠️ SN {sn}: Not enough data available to plot residuals.")
        return
    
    # Extract features and true values
    X = df_sn[features].to_numpy()
    y_true = df_sn[target].to_numpy()

    # Predict and calculate residuals
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    # Plot residuals
    plt.figure(figsize=(12, 4))
    plt.plot(residuals[:n_points], color='red')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"SN {sn} – Residual Plot (Actual - Predicted)")
    plt.xlabel("Sample")
    plt.ylabel("Residual (N)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



metadata_path = 'C:/Users/kingGester/Desktop/data/raw/metadata.csv'
meta = pd.read_csv(metadata_path)

def get_full_path(row):
    folder = 'train' if row['sn'] <= 12 else 'test'
    return f"C:/Users/kingGester/Desktop/data/raw/{folder}/{row['filename']}"

meta['full_path'] = meta.apply(get_full_path, axis=1)

train_data = meta[meta['sn'] .isin([1,2,3])]

# prossec CSV
def process_test_file(file_path, target_column='thrust'):
    if not os.path.exists(file_path):
        print(f"🚨 file {file_path} NOT fonde")
        return None
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            print(f"❌ coulmn {target_column} در فایل {file_path} وجود ندارد.")
            return None

        df = df[['ton', target_column]].copy()
        df.dropna(inplace=True)

        on_duration = []
        count = 0
        for ton in df['ton']:
            count = count + 1 if ton == 1 else 0
            on_duration.append(count)
        df['on_duration'] = on_duration

        df['lag_thrust_1'] = df[target_column].shift(1)
        df.dropna(inplace=True)

        df['source_file'] = file_path
        return df

    except Exception as e:
        print(f"🚨 erore {file_path}: {e}")
        return None

all_train_frames = []
for idx, row in train_data.iterrows():
    file_path = row['full_path']
    sn_value = row['sn']

    df = process_test_file(file_path)
    if df is not None:
        df['sn'] = sn_value
        all_train_frames.append(df)

df_train = pd.concat(all_train_frames, ignore_index=True)

df_train['rolling_avg_thrust'] = df_train.groupby('sn')['thrust'].rolling(window=5).mean().reset_index(drop=True)
df_train['cumulative_on_time'] = df_train.groupby('sn')['on_duration'].cumsum()

features = ['ton', 'on_duration', 'lag_thrust_1', 'rolling_avg_thrust', 'cumulative_on_time']
target = 'thrust'

os.makedirs("models/individual", exist_ok=True)

serial_numbers = df_train['sn'].unique()
results = []

for sn in serial_numbers:
    print(f"\n🔧 Training model for SN: {sn}")

    df_sn = df_train[df_train['sn'] == sn].dropna(subset=features + [target])
    
    if len(df_sn) < 100:
        print(f"⚠️ داده کافی برای {sn} وجود ندارد.")
        continue

    X = df_sn[features].to_numpy()
    y = df_sn[target].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    results.append({
        'SN': sn,
        'MAE': mae,
        'R2': r2
    })

    joblib.dump(model, f"models/individual/xgb_model_{sn}.joblib")
    print(f"✅ SN {sn} | MAE: {mae:.4f} | R²: {r2:.4f}")

results_df = pd.DataFrame(results)
display(results_df.sort_values(by='R2', ascending=False).reset_index(drop=True))
