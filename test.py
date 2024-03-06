import numpy as np
import onnxruntime as ort
import random

def predict_onnx(sv_config_str, flow_sp):
    try:
        session = ort.InferenceSession("rf_model.onnx") #

        sv_values = [int(char) for char in sv_config_str]
        input_data = np.array([sv_values + [flow_sp]], dtype=np.float32)

        # Get model input name
        input_name = session.get_inputs()[0].name

        # Perform prediction
        prediction = session.run(None, {input_name: input_data})

        # Introduce randomness
        perturbation = random.uniform(-0.05, 0.05)
        perturbed_prediction = [value + value * perturbation for value in prediction[0][0]]

        return perturbed_prediction

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

sv_config_str = "000001110" # Take input from each valve then combine to this string
flow_sp = 10
result = predict_onnx(sv_config_str, flow_sp)

if result is not None:
    print(f"Predicted Flow Rate: {result[0]}")
    print(f"Predicted C Valve Percent Open: {result[1]}")
    print(f"Predicted DPT_01: {result[2]}")
    print(f"Predicted DPT_02: {result[3]}")
    print(f"Predicted DPT_03: {result[4]}")
    print(f"Predicted DPT_04: {result[5]}")
    print(f"Predicted DPT_05: {result[6]}")
    print(f"Predicted DPT_06: {result[7]}")
    print(f"Predicted DPT_07: {result[8]}")
    print(f"Predicted DPT_08: {result[9]}")
    print(f"Predicted DPT_09: {result[10]}")
    print(f"Predicted DPT_10: {result[11]}")
    print(f"Predicted DPT_11: {result[12]}")
    print(f"Predicted GPT_01: {result[13]}")
    print(f"Predicted GPT_02: {result[14]}")
    print(f"Predicted GPT_03: {result[15]}")
else:
    print("Error occurred during prediction.")
