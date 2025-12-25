# Hướng dẫn cài đặt môi trường

## Yêu cầu
- Python >= 3.8
- pip

## Cài đặt

### 1. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Cài đặt dependencies
```bash
pip install -r environment/requirements.txt
```

**Lưu ý**: Nếu gặp lỗi "Fatal error in launcher", thử dùng:
```bash
python -m pip install -r environment/requirements.txt
```

### 3. Cấu hình Kaggle API (để download dataset)
- Tạo file `~/.kaggle/kaggle.json` với API credentials từ Kaggle
- Hoặc download dataset thủ công từ: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024

### 4. Chạy Jupyter Notebook
```bash
jupyter notebook
```

## Kiểm tra cài đặt
```bash
python -c "import pandas, numpy, sklearn, xgboost; print('OK')"
```

