# Spacecraft Thruster Performance Prediction | پیش‌بینی عملکرد پیشرانه فضاپیما

## Overview | مرور کلی
This project focuses on analyzing and predicting the performance of spacecraft thrusters using real test data. The dataset contains firing test data from multiple thrusters, including various operating conditions and performance metrics.



## Dataset Source | منبع داده‌ها
The dataset is sourced from Kaggle: [Spacecraft Thruster Firing Tests Dataset](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset)



## Project Structure | ساختار پروژه
```
spacecraft-thruster-prediction/
├── data/
│   ├── raw/           
│   ├── processed/     
│   └── metadata.csv 
|  
├── notebooks/
│   ├── models/
|   |   ├──individual
|   |   └──xgb_final_model.joblib
│   ├── 01_metadata_exploration.ipynb
│   ├── 03_dynamic_features_model.ipynb
│   ├── 04_individualized_modeling.ipynb  
|   ├── 05_lstm_vs_xgboost_SN04.ipynb 
|   └── vl_check.ipynb
├── src/              
└── README.md         
```

## Features | ویژگی‌ها
- Thrust measurement prediction | 
- Anomaly detection in thruster performance |
- Analysis of different operating modes | 
- Performance analysis across different pressure levels | 
- Cumulative throughput and pulse analysis | 

## Data Description | توضیحات داده
The dataset includes: | 
- Multiple thruster serial numbers (SN01-SN24) | 
- Various test pressures (5-24 bars) |
- Different test modes (SSF, ONMOD, OFFMOD, random) | 
- Performance metrics: | 
  - Thrust |
  - Mass flow rate | 
  - Valve states | 
  - Anomaly codes | 
  - Cumulative metrics | 

## Requirements | نیازمندی‌ها
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## download model
To access the ready-made model, follow the steps below.
 ```
│── models/
|   ├──individual
|   └──xgb_final_model.joblib
 ```
xgb_final_model.joblib:  Customization model for all single-propellant chemical propulsion

Inside the folder are models that are customized for their own propulsion engine.


## Installation | نصب
```bash
# Clone the repository
git clone https://github.com/yourusername/spacecraft-thruster-prediction.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage | نحوه استفاده
1. Data Exploration: | کاوش داده‌ها:
```python
# Load metadata
import pandas as pd
metadata = pd.read_csv('data/metadata.csv')

# Load test data
test_data = pd.read_csv('data/raw/train/00001_001_SN01_24bars_ssf.csv')
```

2. Model Training: | آموزش مدل:
```python
# Example code for training a model
from src.models import ThrusterModel
model = ThrusterModel()
model.train(X_train, y_train)
```

## Project Goals | اهداف پروژه
1. Predict thrust performance under various conditions | 
2. Detect anomalies in thruster operation | 
3. Analyze the impact of different operating modes | 
4. Study the relationship between pressure and performance | 
5. Monitor cumulative effects on thruster performance | 

## Contributing | مشارکت
Contributions are welcome! Please feel free to submit a Pull Request.

مشارکت‌ها مورد استقبال قرار می‌گیرند! لطفاً درخواست Pull Request ارسال کنید.

## License | مجوز
This project is licensed under the MIT License - see the LICENSE file for details.

این پروژه تحت مجوز MIT است - برای جزئیات به فایل LICENSE مراجعه کنید.

## Acknowledgments | قدردانی
- Original dataset creators | سازندگان اصلی مجموعه داده
- Kaggle community | جامعه کگل
- Contributors and maintainers | مشارکت‌کنندگان و نگهدارندگان
#