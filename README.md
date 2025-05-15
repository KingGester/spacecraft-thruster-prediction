# Spacecraft Thruster Performance Prediction | پیش‌بینی عملکرد پیشرانه فضاپیما

## Overview | مرور کلی
This project focuses on analyzing and predicting the performance of spacecraft thrusters using real test data. The dataset contains firing test data from multiple thrusters, including various operating conditions and performance metrics.

این پروژه بر تحلیل و پیش‌بینی عملکرد پیشرانه‌های فضاپیما با استفاده از داده‌های واقعی تست تمرکز دارد. مجموعه داده شامل اطلاعات تست شلیک از چندین پیشرانه، با شرایط عملیاتی و معیارهای عملکرد مختلف است.

## Dataset Source | منبع داده‌ها
The dataset is sourced from Kaggle: [Spacecraft Thruster Firing Tests Dataset](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset)

مجموعه داده از کگل گرفته شده است: خفن[مجموعه داده تست شلیک پیشرانه فضاپیما](https://www.kaggle.com/datasets/patrickfleith/spacecraft-thruster-firing-tests-dataset)

## Project Structure | ساختار پروژه
```
spacecraft-thruster-prediction/
├── data/
│   ├── raw/           
│   ├── processed/     
│   └── metadata.csv 
|  
├── notebooks/         
├── src/              
└── README.md         
```

## Features | ویژگی‌ها
- Thrust measurement prediction | پیش‌بینی اندازه‌گیری رانش
- Anomaly detection in thruster performance | تشخیص ناهنجاری در عملکرد پیشرانه
- Analysis of different operating modes | تحلیل حالت‌های مختلف عملیاتی
- Performance analysis across different pressure levels | تحلیل عملکرد در سطوح مختلف فشار
- Cumulative throughput and pulse analysis | تحلیل تجمعی توان عملیاتی و پالس

## Data Description | توضیحات داده
The dataset includes: | مجموعه داده شامل موارد زیر است:
- Multiple thruster serial numbers (SN01-SN24) | شماره سریال‌های متعدد پیشرانه
- Various test pressures (5-24 bars) | فشارهای تست مختلف
- Different test modes (SSF, ONMOD, OFFMOD, random) | حالت‌های تست مختلف
- Performance metrics: | معیارهای عملکرد:
  - Thrust | رانش
  - Mass flow rate | نرخ جریان جرمی
  - Valve states | وضعیت شیرها
  - Anomaly codes | کدهای ناهنجاری
  - Cumulative metrics | معیارهای تجمعی

## Requirements | نیازمندی‌ها
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

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
1. Predict thrust performance under various conditions | پیش‌بینی عملکرد رانش در شرایط مختلف
2. Detect anomalies in thruster operation | تشخیص ناهنجاری‌ها در عملکرد پیشرانه
3. Analyze the impact of different operating modes | تحلیل تأثیر حالت‌های مختلف عملیاتی
4. Study the relationship between pressure and performance | مطالعه رابطه بین فشار و عملکرد
5. Monitor cumulative effects on thruster performance | نظارت بر تأثیرات تجمعی بر عملکرد پیشرانه

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
