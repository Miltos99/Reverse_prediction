Reverse Prediction Project
Overview
This project aims to predict one input based on five output variables using various machine learning and deep learning techniques. The provided code includes implementations for Random Forest, LightGBM, XGBoost, LSTM, Conv1D, and hybrid models.

Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, scikit-learn, xgboost, lightgbm, tensorflow, keras
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/reverse-prediction-project.git
Navigate to the project directory:
bash
Copy code
cd reverse-prediction-project
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
Run the main script:
bash
Copy code
python main.py
Choose the desired model type when prompted.
Models
RandomForest: Random Forest regression model.
LightGBM: Light Gradient Boosting Machine regression model.
XGBoost: eXtreme Gradient Boosting regression model.
XGBoost_RandomSearch: XGBoost regression model with hyperparameter tuning using Random Search.
RandomForest_hyperparam: Random Forest regression model with hyperparameter tuning using RandomizedSearchCV.
SingleTarget: Individual Random Forest models for each output variable.
Rnd_For_Reg: Random Forest regression model with hyperparameter tuning using RandomizedSearchCV.
LSTM_Hybrid: LSTM-based hybrid model.
CONV1d_Hybrid: Conv1D-based hybrid model.
Data
The project assumes input data in Excel format. Adjust the data_path variable in the main.py file to point to your dataset.

License
This project is licensed under the MIT License - see the LICENSE file for details.
