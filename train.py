import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm  # 用於載入更多預訓練模型
from tqdm import tqdm
import time
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # 簡化日誌格式
    handlers=[
        logging.StreamHandler()  # 只保留控制台輸出
    ]
)
logger = logging.getLogger(__name__)

# 設置設備
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logger.warning("未偵測到 GPU，將使用 CPU 進行訓練。若需加速，請安裝 CUDA 並使用支援的顯示卡。")

# 可選：強制要求必須有 GPU
# assert torch.cuda.is_available(), "未偵測到 GPU，請確認 CUDA 驅動與顯示卡安裝正確！"

class PhishIRISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"無法載入圖片 {img_path}: {str(e)}")
            # 返回一個空白圖片
            return torch.zeros(3, 224, 224), label

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(y_true, y_pred):
    """計算評估指標"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = (y_true == y_pred).mean()
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=10)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 顯示初始資訊
    logger.info(f'開始訓練 | 總 Epochs: {num_epochs}')
    logger.info('='*50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 訓練階段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [訓練]', leave=False)
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            
            # 收集預測結果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{torch.sum(preds == labels.data).item()/batch_size:.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 計算訓練指標
        train_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        # 驗證階段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_val_preds = []
        all_val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [驗證]', leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                # 收集驗證預測結果
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).item()/batch_size:.4f}'
                })
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 計算驗證指標
        val_metrics = calculate_metrics(np.array(all_val_labels), np.array(all_val_preds))
        
        # 計算 epoch 時間
        epoch_time = time.time() - epoch_start_time
        
        # 輸出詳細的訓練資訊
        logger.info(f'Epoch {epoch+1}/{num_epochs} | '
                   f'訓練: {epoch_loss:.4f}/{epoch_acc:.4f} | '
                   f'驗證: {val_loss:.4f}/{val_acc:.4f} | '
                   f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                   f'時間: {epoch_time:.1f}s')
        
        # 輸出詳細指標
        logger.info(f'訓練指標: '
                   f'準確率={train_metrics["accuracy"]:.4f} | '
                   f'精確度={train_metrics["precision"]:.4f} | '
                   f'召回率={train_metrics["recall"]:.4f} | '
                   f'F1={train_metrics["f1"]:.4f}')
        logger.info(f'驗證指標: '
                   f'準確率={val_metrics["accuracy"]:.4f} | '
                   f'精確度={val_metrics["precision"]:.4f} | '
                   f'召回率={val_metrics["recall"]:.4f} | '
                   f'F1={val_metrics["f1"]:.4f}')
        
        # 更新學習率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_metrics['f1'] > best_acc:  # 使用 F1 分數作為保存標準
            best_acc = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metrics': val_metrics,
                'history': history
            }, 'models/best_model.pth')
            logger.info(f'✓ 已保存最佳模型 (F1: {best_acc:.4f})')
        
        # 早停檢查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("⚠️ 觸發早停機制")
            break
    
    # 簡化最終輸出
    logger.info('='*50)
    logger.info(f'訓練完成 | 最佳 F1 分數: {best_acc:.4f}')

def main():
    # 簡化初始化輸出
    logger.info('初始化訓練環境...')
    
    # 增強資料轉換
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 載入資料集
    train_dataset = PhishIRISDataset(
        root_dir='data/phishIRIS_DL_Dataset/train',
        transform=data_transforms['train']
    )
    
    val_dataset = PhishIRISDataset(
        root_dir='data/phishIRIS_DL_Dataset/val',
        transform=data_transforms['val']
    )
    
    logger.info(f'資料集: 訓練 {len(train_dataset)} 張 | 驗證 {len(val_dataset)} 張')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = timm.create_model('efficientnet_b3', pretrained=True)
    num_classes = len(train_dataset.classes)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    model = model.to(device)
    
    logger.info(f'模型: EfficientNet-B3 | 類別數: {num_classes}')
    
    # 使用標籤平滑的交叉熵損失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 使用 AdamW 優化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 使用學習率調度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 訓練模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)
    
    # 保存類別映射
    class_mapping = {i: cls for i, cls in enumerate(train_dataset.classes)}
    torch.save(class_mapping, 'models/class_mapping.pth')
    logger.info('已保存類別映射')

if __name__ == '__main__':
    main() 