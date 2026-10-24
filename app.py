import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import json
import pandas as pd
from skimage.feature import local_binary_pattern, hog

st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Panel Defect Detection & Diagnostic Dashboard")
st.markdown("### Benchmarking Classical Machine Learning vs Fine-Tuned Deep Learning")

class CustomSolarCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomSolarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_data
def load_metrics():
    try:
        with open("metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "ResNet50": {"acc": 0.9138, "f1": 0.9130},
            "Ensemble": {"acc": 0.9023, "f1": 0.9019},
            "EfficientNet": {"acc": 0.8793, "f1": 0.8780},
            "SVM": {"acc": 0.7184, "f1": 0.7124},
            "XGBoost": {"acc": 0.6954, "f1": 0.6895},
            "Random Forest": {"acc": 0.6724, "f1": 0.6628},
            "Custom CNN": {"acc": 0.5115, "f1": 0.4618}
        }

metrics = load_metrics()

@st.cache_resource
def load_all_artifacts():
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    svm_model = joblib.load("svm_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    
    num_classes = len(encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet = models.resnet50(weights=None)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes))
    resnet.load_state_dict(torch.load("resnet50_fine_tuned.pth", map_location=device))
    resnet.to(device).eval()
    
    effnet = models.efficientnet_b0(weights=None)
    num_ftrs_eff = effnet.classifier[1].in_features
    effnet.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs_eff, num_classes))
    effnet.load_state_dict(torch.load("efficientnet_b0_fine_tuned.pth", map_location=device))
    effnet.to(device).eval()
    
    custom_cnn = CustomSolarCNN(num_classes=num_classes)
    custom_cnn.load_state_dict(torch.load("custom_cnn_solar.pth", map_location=device))
    custom_cnn.to(device).eval()
    
    return scaler, encoder, svm_model, xgb_model, rf_model, resnet, effnet, custom_cnn, device

scaler, encoder, svm_model, xgb_model, rf_model, resnet, effnet, custom_cnn, device = load_all_artifacts()

def extract_classical_features(img_rgb):
    img_resized = cv2.resize(img_rgb, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_feat = hist.flatten()
    
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
    
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    combined = np.hstack([color_feat, lbp_hist, hog_feat])
    return scaler.transform([combined])

st.sidebar.header("🕹️ Model Selection")

model_options = {
    f"🏆 Fine-Tuned ResNet50 ({metrics['ResNet50']['acc']:.2%} Acc)": "ResNet50",
    f"🤝 Soft Ensemble ({metrics['Ensemble']['acc']:.2%} Acc)": "Ensemble",
    f"⚡ Fine-Tuned EfficientNet-B0 ({metrics['EfficientNet']['acc']:.2%} Acc)": "EfficientNet",
    f"🎯 Tuned SVM ({metrics['SVM']['acc']:.2%} Acc)": "SVM",
    f"🌲 Tuned XGBoost ({metrics['XGBoost']['acc']:.2%} Acc)": "XGBoost",
    f"🌳 Tuned Random Forest ({metrics['Random Forest']['acc']:.2%} Acc)": "Random Forest",
    f"🧱 Custom CNN Scratch ({metrics['Custom CNN']['acc']:.2%} Acc)": "Custom CNN"
}

selected_label = st.sidebar.selectbox("Choose Model Architecture:", list(model_options.keys()))
model_choice = model_options[selected_label]

uploaded_file = st.sidebar.file_uploader("Upload Inspection Image...", type=["jpg", "jpeg", "png"])

tab1, tab2 = st.tabs(["🔍 Live Inspection", "📊 Benchmark Leaderboard"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        with col1:
            st.subheader("Uploaded Solar Panel Scan")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Diagnostic Prediction")
            
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            if model_choice == "ResNet50":
                input_tensor = tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(resnet(input_tensor), dim=1).cpu().numpy()[0]
            elif model_choice == "Ensemble":
                input_tensor = tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    p1 = torch.softmax(resnet(input_tensor), dim=1)
                    p2 = torch.softmax(effnet(input_tensor), dim=1)
                    probs = ((p1 + p2) / 2.0).cpu().numpy()[0]
            elif model_choice == "EfficientNet":
                input_tensor = tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(effnet(input_tensor), dim=1).cpu().numpy()[0]
            elif model_choice == "Custom CNN":
                input_tensor = tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(custom_cnn(input_tensor), dim=1).cpu().numpy()[0]
            else:
                features = extract_classical_features(img_np)
                if model_choice == "SVM":
                    probs = svm_model.predict_proba(features)[0]
                elif model_choice == "XGBoost":
                    probs = xgb_model.predict_proba(features)[0]
                else:
                    probs = rf_model.predict_proba(features)[0]
                    
            pred_idx = np.argmax(probs)
            pred_label = encoder.classes_[pred_idx]
            confidence = probs[pred_idx]
            
            st.success(f"**Predicted Condition:** `{pred_label}`")
            st.info(f"**Confidence Score:** `{confidence:.2%}`")
            
            st.markdown("#### Class Confidence Probability Distribution")
            st.bar_chart({"Class": encoder.classes_, "Probability": probs}, x="Class", y="Probability")
    else:
        st.info("👈 Upload a solar panel image from the sidebar to inspect defects live.")

with tab2:
    st.subheader("Model Performance Leaderboard")
    
    table_data = []
    for model_key, stats in metrics.items():
        category = "Deep Learning" if model_key in ["ResNet50", "Ensemble", "EfficientNet", "Custom CNN"] else "Classical ML"
        table_data.append({
            "Model Name": model_key,
            "Category": category,
            "Accuracy": f"{stats['acc']:.2%}",
            "Weighted F1-Score": f"{stats['f1']:.2%}"
        })
        
    df_metrics = pd.DataFrame(table_data).sort_values(by="Accuracy", ascending=False)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)