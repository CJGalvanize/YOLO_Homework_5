import torch
import sys
import os
import pandas as pd
import numpy as np
from sahi.predict import get_sliced_prediction
from sahi.models.base import DetectionModel
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Load the weights
native_model = attempt_load("y7_best.pt", map_location="cpu")

# Define the SAHI Wrapper (The Custom Bridge)
class YOLOv7SahiWrapper(DetectionModel):
    def load_model(self):
        self._is_loaded = True

    def perform_inference(self, image: any):
        img_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            prediction = self.model(img_tensor)
        nms_results = non_max_suppression(prediction[0], conf_thres=0.25, iou_thres=0.45)
        self._original_predictions = nms_results[0]
        print(f"  -> Detections after NMS: {len(self._original_predictions)}")

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list, full_shape_list, **kwargs
    ):
        from sahi.prediction import ObjectPrediction

        object_prediction_list = []

        for x in self._original_predictions:
            if x[4] > self.confidence_threshold:
                bbox = [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=int(x[5]),
                    score=float(x[4]),
                    category_name=str(int(x[5])),
                    shift_amount=shift_amount_list,
                    full_shape=full_shape_list
                )
                object_prediction_list.append(object_prediction)
        self._object_prediction_list = object_prediction_list

# Initialize SAHI model
detection_model = YOLOv7SahiWrapper(model_path=None, confidence_threshold=0.25, device="cpu")
detection_model.model = native_model
detection_model._is_loaded = True

# Run Inference
images = ["ND_6m_0030.JPG", "ND_6m_0033.JPG", "ND_6m_0009.JPG", "FN_6m_0030.JPG", "FN_6m_0009.JPG", "FN_6m_0033.JPG"]
all_predictions = []

for img_name in images:
    if not os.path.exists(img_name):
        print(f"Skipping {img_name} - File not found.")
        continue
    print(f"Processing {img_name}...")
    result = get_sliced_prediction(img_name, detection_model, slice_height=640, slice_width=640)

    for p in result.object_prediction_list:
        all_predictions.append({
            "image_id": img_name,
            "category_id": p.category_id,
            "confidence": p.score.value,
            "bbox": p.bbox.to_xyxy(),
        })

pd.DataFrame(all_predictions).to_csv("homework_5_results.csv", index=False)
print("Success! Results saved to homework_5_results.csv")