# Hướng dẫn Deploy Streamlit App

## Phương pháp 1: Streamlit Cloud (Khuyến nghị - Miễn phí)

### Bước 1: Chuẩn bị
1. Đảm bảo đã train model và có các file:
   - `src/models/xgboost_model.pkl`
   - `src/models/preprocessor.pkl`
   - `src/models/xgboost_model.json`
   - `data/raw/vietnam_housing_dataset.csv`

2. Đảm bảo có file `requirements.txt` ở thư mục gốc

3. Đảm bảo có file `streamlit_app.py` ở thư mục gốc (file entry point)

### Bước 2: Push code lên GitHub
```bash
git init
git add .
git commit -m "Initial commit for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Bước 3: Deploy lên Streamlit Cloud
1. Truy cập: https://share.streamlit.io/
2. Đăng nhập bằng GitHub
3. Click "New app"
4. Chọn repository và branch
5. Main file path: `streamlit_app.py`
6. Click "Deploy"

### Bước 4: Cấu hình (nếu cần)
- App sẽ tự động detect `requirements.txt`
- Nếu cần thay đổi config, sửa file `.streamlit/config.toml`

---

## Phương pháp 2: Heroku

### Bước 1: Tạo các file cần thiết

**Procfile** (tạo ở thư mục gốc):
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh** (tạo ở thư mục gốc):
```bash
mkdir -p ~/.streamlit

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml
```

### Bước 2: Deploy
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## Phương pháp 3: Docker + VPS/Cloud

### Bước 1: Tạo Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Bước 2: Build và chạy
```bash
docker build -t housing-prediction-app .
docker run -p 8501:8501 housing-prediction-app
```

---

## Phương pháp 4: Railway (Miễn phí với giới hạn)

1. Đăng ký tại: https://railway.app/
2. Connect GitHub repository
3. Railway tự động detect và deploy
4. Cấu hình PORT nếu cần

---

## Phương pháp 5: Render

1. Đăng ký tại: https://render.com/
2. Tạo Web Service mới
3. Connect GitHub repository
4. Cấu hình:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

---

## Lưu ý quan trọng:

1. **File size**: Streamlit Cloud có giới hạn 1GB. Nếu model quá lớn, cần:
   - Compress model
   - Hoặc dùng Git LFS
   - Hoặc host model trên cloud storage (S3, GCS) và load từ URL

2. **Dữ liệu nhạy cảm**: Không commit API keys, passwords vào Git

3. **Performance**: 
   - Sử dụng `@st.cache_resource` cho model loading
   - Sử dụng `@st.cache_data` cho data loading

4. **Testing**: Test app local trước khi deploy:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Troubleshooting:

### Lỗi: Module not found
- Kiểm tra `requirements.txt` có đủ dependencies
- Kiểm tra import paths trong code

### Lỗi: Model file not found
- Đảm bảo model files được commit vào Git
- Kiểm tra đường dẫn trong `streamlit_app.py`

### App chạy chậm
- Optimize model size
- Sử dụng caching đúng cách
- Consider using lighter model for production

