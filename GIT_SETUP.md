# Hướng dẫn Push Model Files lên GitHub

## Vấn đề
File `.pkl` (model files) bị ignore trong `.gitignore`, cần force add để commit.

## Giải pháp

### Bước 1: Force add các file model
```bash
cd d:\temp\project
git add -f src/models/xgboost_model.pkl
git add -f src/models/preprocessor.pkl
git add -f src/models/xgboost_model.json
```

### Bước 2: Kiểm tra file đã được add
```bash
git status
```

### Bước 3: Commit và push
```bash
git commit -m "Add model files for deployment"
git push origin main
```

## Lưu ý về kích thước file

Nếu file model quá lớn (>100MB), GitHub sẽ từ chối. Có 2 cách xử lý:

### Cách 1: Sử dụng Git LFS (Large File Storage)
```bash
# Cài đặt Git LFS
git lfs install

# Track các file .pkl
git lfs track "*.pkl"
git lfs track "src/models/*.pkl"

# Add và commit
git add .gitattributes
git add src/models/*.pkl
git commit -m "Add model files with Git LFS"
git push origin main
```

### Cách 2: Host model trên cloud storage
- Upload model lên Google Drive / Dropbox / S3
- Sửa code để download model khi app khởi động
- Hoặc dùng environment variable để chỉ định URL model

## Kiểm tra sau khi push

1. Vào GitHub repository
2. Kiểm tra file `src/models/xgboost_model.pkl` có tồn tại không
3. Nếu có, Streamlit Cloud sẽ tự động detect và deploy

## Troubleshooting

### Lỗi: File quá lớn
- Sử dụng Git LFS
- Hoặc compress model trước khi commit
- Hoặc host model riêng

### Lỗi: File vẫn bị ignore
- Kiểm tra `.gitignore` đã có `!src/models/*.pkl` chưa
- Dùng `git add -f` để force add

