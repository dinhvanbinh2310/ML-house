# Demo App - Dự đoán Giá Nhà

Ứng dụng web demo để dự đoán giá nhà sử dụng mô hình XGBoost đã được train.

## Yêu cầu

- Model đã được train (file `src/models/xgboost_model.pkl`)
- Preprocessor đã được lưu (file `src/models/preprocessor.pkl`)

## Cài đặt

```bash
pip install streamlit
```

Hoặc cài từ requirements.txt:
```bash
pip install -r environment/requirements.txt
```

## Chạy ứng dụng

```bash
cd project
streamlit run src/app/app.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

## Sử dụng

1. Nhập thông tin nhà vào form
2. Click nút "Dự đoán Giá"
3. Xem kết quả dự đoán

## Lưu ý

- Cần train model trước khi sử dụng app
- Một số trường có thể để trống (sẽ được xử lý tự động)
- Giá dự đoán tính bằng tỷ VNĐ

