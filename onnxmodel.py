import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# Load data
df = pd.read_csv('Lab19_Dataset.csv')

# Prepare data
X = pd.get_dummies(df[['SV_01', 'SV_02', 'SV_03', 'SV_04', 'SV_05', 'SV_06', 'SV_07', 'SV_08', 'SV_09', 'Flow SP (GPM)']])
y = df[['Flow Rate (GPM)', 'C_Valve %Open', 'DPT_01 (PSI)', 'DPT_02 (PSI)', 'DPT_03 (PSI)', 'DPT_04 (PSI)',
        'DPT_05 (PSI)', 'DPT_06 (PSI)', 'DPT_07 (PSI)', 'DPT_08 (PSI)', 'DPT_09 (PSI)', 'DPT_10 (PSI)',
        'DPT_11 (PSI)', 'GPT_01 (PSI)', 'GPT_02 (PSI)', 'GPT_03 (PSI)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Define input and output types
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# Convert model to ONNX format
onx = convert_sklearn(rf_model, initial_types=initial_type)

# Save ONNX model to file
onnx.save_model(onx, "rf_model.onnx")
