import torch
import timm

# Path to your .bin file
MODEL_PATH = "C:/Users/joana/Downloads/fundus_diabetes_retinopathy/best_model_kappa_512.pth"

# Recreate architecture
model = timm.create_model('efficientnet_b6', pretrained=False, num_classes=1)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Handle different save formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Export
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model,
    dummy_input,
    "dr_model.onnx",
    input_names=['image'],
    output_names=['severity_score'],
    dynamic_axes={'image': {0: 'batch'}, 'severity_score': {0: 'batch'}},
    opset_version=17
)

print("Exported to dr_model.onnx")