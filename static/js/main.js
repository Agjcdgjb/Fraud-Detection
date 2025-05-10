document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const loading = document.getElementById('loading');
    const resultClass = document.getElementById('resultClass');
    const resultConfidence = document.getElementById('resultConfidence');

    // 點擊上傳區域觸發檔案選擇
    dropZone.addEventListener('click', () => fileInput.click());

    // 處理檔案選擇
    fileInput.addEventListener('change', handleFileSelect);

    // 處理拖放
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 移除圖片
    removeBtn.addEventListener('click', () => {
        previewContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        fileInput.value = '';
        dropZone.style.display = 'block';
    });

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // 檢查檔案類型
        if (!file.type.match('image.*')) {
            showError('請上傳圖片檔案！');
            return;
        }

        // 檢查檔案大小（16MB）
        if (file.size > 16 * 1024 * 1024) {
            showError('檔案大小不能超過 16MB！');
            return;
        }

        // 顯示預覽
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            resultContainer.style.display = 'none';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // 上傳檔案
        uploadFile(file);
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        loading.style.display = 'block';
        resultContainer.style.display = 'none';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('網路錯誤');
            }
            return response.json();
        })
        .then(data => {
            loading.style.display = 'none';
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // 顯示結果
            resultClass.textContent = data.class;
            resultConfidence.textContent = data.confidence;
            resultContainer.style.display = 'block';
            
            // 根據結果添加不同的樣式
            const confidence = parseFloat(data.confidence);
            if (confidence >= 80) {
                resultContainer.classList.add('high-confidence');
            } else if (confidence >= 50) {
                resultContainer.classList.add('medium-confidence');
            } else {
                resultContainer.classList.add('low-confidence');
            }
        })
        .catch(error => {
            loading.style.display = 'none';
            showError('上傳失敗，請重試！');
            console.error('Error:', error);
        });
    }

    function showError(message) {
        // 創建錯誤提示元素
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // 添加到頁面
        document.body.appendChild(errorDiv);
        
        // 3秒後自動移除
        setTimeout(() => {
            errorDiv.remove();
        }, 3000);
    }
}); 