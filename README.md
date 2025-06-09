# 詐騙圖片檢測系統

這是一個基於深度學習的詐騙圖片檢測系統，使用 Flask 框架建立的網頁應用程式。系統可以自動分析上傳的圖片，並判斷其是否為詐騙圖片。

## 功能特點

- 支援圖片上傳（JPG、PNG 格式）
- 即時圖片分析
- 顯示檢測結果和可信度
- 響應式設計，支援行動裝置
- 直覺的使用者介面
- 支援拖放上傳

## 系統需求

- Python 3.8 或更高版本
- CUDA 支援（建議用於 GPU 加速）
- 至少 4GB RAM
- 現代網頁瀏覽器

## 安裝步驟

1. 克隆專案：
```bash
git clone [專案網址]
cd [專案目錄]
```

2. 建立並啟動虛擬環境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

4. 確保模型檔案存在：
- 將訓練好的模型檔案放在 `models/best_model.pth`
- 將類別映射檔案放在 `models/class_mapping.pth`

## 使用方法

1. 啟動應用程式：
```bash
python app.py
```

2. 開啟網頁瀏覽器，訪問：
```
http://localhost:5000
```

3. 上傳圖片：
- 點擊上傳區域選擇圖片
- 或直接拖放圖片到上傳區域

4. 查看結果：
- 系統會自動分析圖片
- 顯示檢測結果和可信度

## 專案結構

```
├── app.py              # Flask 應用程式主檔案
├── train.py           # 模型訓練腳本
├── models/            # 模型檔案目錄
│   ├── best_model.pth # 訓練好的模型
│   └── class_mapping.pth # 類別映射檔案
├── static/           # 靜態檔案目錄
│   ├── css/         # CSS 樣式檔案
│   ├── js/          # JavaScript 檔案
│   └── images/      # 圖片資源
├── templates/        # HTML 模板目錄
├── data/            # 訓練數據目錄
├── uploads/         # 上傳檔案暫存目錄
├── venv/            # Python 虛擬環境
└── requirements.txt  # 依賴套件清單
```

## 技術架構

- 後端：Flask 2.2.0+
- 前端：HTML5、CSS3、JavaScript
- 深度學習：PyTorch 2.0.1
- 模型架構：EfficientNet-B3
- 圖片處理：Pillow 9.0.0+
- 其他依賴：
  * timm 0.5.4（模型架構）
  * scikit-learn 0.24.0+（數據處理）
  * beautifulsoup4（網頁解析）

## 注意事項

- 上傳圖片大小限制為 16MB
- 支援的圖片格式：JPG、PNG
- 建議使用支援 CUDA 的 GPU 以獲得更好的效能
- 系統會自動記錄檢測結果以改進模型效能
- 建議定期更新依賴套件以修復安全漏洞