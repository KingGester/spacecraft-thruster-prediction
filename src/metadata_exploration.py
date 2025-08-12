import pandas as pd
import matplotlib.pyplot as plt
# read a metadata 
metadata_path = 'C:/Users/kingGester/Desktop/data/raw/metadata.csv'
meta=pd.read_csv(metadata_path)

meta['sn_str'] = meta['sn'].apply(lambda x: f"SN{str(x).zfill(2)}")


def get_full_path(row):
    """ Each row gets a full file path stored in meta['full_path'],
    pointing to either the train or test directory based on the 'sn' value
    
    Returns:
        string: full file path
    """

    folder = 'train' if row['sn'] <= 12 else 'test'
    return f"C:/Users/kingGester/Desktop/data/raw/{folder}/{row['filename']}"



    
def process_test_file(file_path, target_column='thrust'):
    """give a data file test and split columns and return 

    Args:
        file_path (_type_): _description_
        target_column (str, optional): _description_. Defaults to 'thrust'.

    Returns:
        df: data file path
    """
    if not os.path.exists(file_path):
        print(f"🚨 file {file_path} not find")
        return None
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            print(f"❌ columns{target_column} in the file{file_path} not here ")
            return None

        df = df[['ton', target_column]].copy()
        df.dropna(inplace=True)

        # Calculator on_duration
        on_duration = []
        count = 0
        for ton in df['ton']:
            if ton == 1:
                count += 1
            else:
                count = 0
            on_duration.append(count)
        df['on_duration'] = on_duration

        df['lag_thrust_1'] = df[target_column].shift(1)
        df.dropna(inplace=True)  

        df['source_file'] = file_path
        return df

    except Exception as e:
        print(f"🚨 Error in file{file_path}: {e}")
        return None


train_data=meta[meta['sn'] <= 6] 
all_train_frames = []
# processing data train
for idx, row in train_data.iterrows():
    file_path = row['full_path']
    df = process_test_file(file_path, target_column='thrust')
    if df is not None:
        all_train_frames.append(df)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
# data (test and train) split
X = df_train[['ton', 'on_duration', 'lag_thrust_1']]
y = df_train['thrust']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#modeling
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("📊 MAE:", mean_absolute_error(y_val, y_pred))
print("📈 R2 Score:", r2_score(y_val, y_pred))



# plt Thrust: Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_val.values[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.legend()
plt.title("Thrust: Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Thrust")
plt.grid(True)
plt.show()


# step by step 
#test_data = meta[(meta['sn'] >= 13) & (meta['sn'] <= 21)] becase we not have larg RAM 

test_data = meta[(meta['sn'] >= 13) & (meta['sn'] <= 14)]
# processing data for test

all_test_frames = []

for idx, row in test_data.iterrows():
    file_path = row['full_path']
    df = process_test_file(file_path, target_column='thrust')
    if df is not None:
        all_test_frames.append(df)


X_test = df_test[['ton', 'on_duration', 'lag_thrust_1']]
y_test = df_test['thrust']

y_test_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

print("📊 MAE (Test Set):", mean_absolute_error(y_test, y_test_pred))
print("📈 R2 Score (Test Set):", r2_score(y_test, y_test_pred))