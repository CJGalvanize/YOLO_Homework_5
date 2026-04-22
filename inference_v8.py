import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# Load the model
#YOLOv8 handles large images by slicing them into smaller pieces, running inference on each piece, and then combining the results.
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.25,
    device="cpu"
)
# Define the input image path
images = ["ND_6m_0030.JPG", "ND_6m_0033.JPG", "ND_6m_0009.JPG", "FN_6m_0030.JPG", "FN_6m_0009.JPG", "FN_6m_0033.JPG"]
# Run inference on each imagepyt
for img_name in images:
    print(f"Processing {img_name}...")
    # Get the sliced prediction for the image
    result = get_sliced_prediction(
        img_name, 
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2, # 20% overlap to ensure objects near the edges are detected
    )
    # Print the output results
    result.export_visuals(export_dir="output/", file_name=f"pred_{img_name}") 
print("Inference completed.")

