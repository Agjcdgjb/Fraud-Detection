import os
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import timm
import logging
import gdown
import tempfile

app = Flask(__name__)

# 修改上傳資料夾路徑為 Heroku 的臨時目錄
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 確保上傳資料夾存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Google Drive 檔案 ID
MODEL_FILE_ID = '12kigRCky5Kjg3XvurmOnjt3miA79pYvi'
CLASS_MAPPING_FILE_ID = '1E7JiFLdFF957dP0-vbDg0wjYxOspPh80'

def download_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# 取得模型與類別映射的本地路徑（自動下載）
def get_model_and_mapping():
    # 在 Heroku 中使用臨時目錄
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, 'best_model.pth')
    class_mapping_path = os.path.join(temp_dir, 'class_mapping.pth')
    download_from_drive(MODEL_FILE_ID, model_path)
    download_from_drive(CLASS_MAPPING_FILE_ID, class_mapping_path)
    return model_path, class_mapping_path

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用設備: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 載入模型和類別映射
def load_model():
    model_path, class_mapping_path = get_model_and_mapping()
    model = timm.create_model('efficientnet_b3', pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.classifier.in_features, 15)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    class_mapping = torch.load(class_mapping_path, map_location=device)
    return model, class_mapping

model, class_mapping = load_model()

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        result = {
            'class': class_mapping[predicted_class],
            'confidence': f"{confidence * 100:.2f}%"
        }
        return result
    except Exception as e:
        logger.error(f"預測時發生錯誤: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = predict_image(filepath)
        os.remove(filepath)
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': '預測失敗'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 