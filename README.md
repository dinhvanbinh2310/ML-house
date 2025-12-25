# Đồ án Khai thác Dữ liệu - Vietnam Housing Dataset 2024

## Mô tả
Dự án phân tích và dự đoán giá nhà tại Việt Nam sử dụng dataset từ Kaggle.

## Dataset
- **Nguồn**: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024
- **Yêu cầu**: >= 5 cột, >= 500 dòng

## Cấu trúc dự án
```
project/
  data/raw/              # Dữ liệu thô
  data/processed/         # Dữ liệu đã xử lý
  src/preprocess/        # Scripts tiền xử lý
  src/models/            # Mô hình ML
  src/evaluation/        # Đánh giá mô hình
  src/utils/             # Utilities
  src/app/               # Demo app (tùy chọn)
  src/main.ipynb         # Notebook chính
  report/draft/          # Báo cáo draft
  report/final/          # Báo cáo PDF cuối
  slides/                # Slides thuyết trình
  environment/           # Cấu hình môi trường
```

## Quy trình

1. **Kiểm tra Dataset**: Thống kê mô tả, kiểm tra chất lượng
2. **Tiền xử lý**: Xử lý missing, outlier, encoding, scaling
3. **Mô hình**: XGBoost với hyperparameter tuning
4. **Đánh giá**: Metrics phù hợp (RMSE, MAE, MAPE, R2 cho regression)
5. **Báo cáo**: PDF và slides

## Cài đặt
Xem chi tiết tại `environment/install.md`

## Chạy dự án

### Notebook chính
```bash
jupyter notebook src/main.ipynb
```

### Demo App (Streamlit)
```bash
cd project
streamlit run src/app/app.py
```

Xem hướng dẫn chi tiết tại `src/app/README.md`

