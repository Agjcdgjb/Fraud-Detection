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
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用設備: {device}")

# 模型和類別映射的雲端 URL
MODEL_URL = "https://drive.google.com/uc?id=12kigRCky5Kjg3XvurmOnjt3miA79pYvi"
CLASS_MAPPING_URL = "https://drive.google.com/uc?id=1E7JiFLdFF957dP0-vbDg0wjYxOspPh80"

def download_from_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        logger.error(f"下載檔案時發生錯誤: {str(e)}")
        return False

# 載入模型和類別映射
def load_model():
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'best_model.pth')
    class_mapping_path = os.path.join(temp_dir, 'class_mapping.pth')
    
    # 下載模型和類別映射
    if not download_from_drive(MODEL_URL, model_path):
        raise Exception("無法下載模型檔案")
    if not download_from_drive(CLASS_MAPPING_URL, class_mapping_path):
        raise Exception("無法下載類別映射檔案")
    
    # 載入模型
    model = timm.create_model('efficientnet_b3', pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.classifier.in_features, 15)  # 15 個類別
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 載入類別映射
    class_mapping = torch.load(class_mapping_path, map_location=device)
    
    # 清理臨時檔案
    os.remove(model_path)
    os.remove(class_mapping_path)
    os.rmdir(temp_dir)
    
    return model, class_mapping

# 載入模型和類別映射
model, class_mapping = load_model()

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        # 載入和轉換圖片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 進行預測
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 獲取預測結果
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
        
        # 進行預測
        result = predict_image(filepath)
        
        # 刪除上傳的檔案
        os.remove(filepath)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': '預測失敗'}), 500

if __name__ == '__main__':
    app.run(debug=True) 