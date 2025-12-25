import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from PIL import Image

sys.path.append(str(Path(__file__).parent / "src"))
from preprocess.preprocess import DataPreprocessor

st.set_page_config(
    page_title="Dự đoán Giá Nhà Việt Nam",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Ứng dụng Dự đoán Giá Nhà Việt Nam")
st.markdown("Nhập thông tin nhà để dự đoán giá")

base_dir = Path(__file__).parent
model_path = base_dir / "src" / "models" / "xgboost_model.pkl"
preprocessor_path = base_dir / "src" / "models" / "preprocessor.pkl"
data_path = base_dir / "data" / "raw" / "vietnam_housing_dataset.csv"

@st.cache_data
def load_addresses():
    try:
        if data_path.exists():
            df = pd.read_csv(data_path)
            addresses = sorted(df['Address'].dropna().unique().tolist())
            return addresses
        return []
    except Exception as e:
        return []

@st.cache_resource
def load_model():
    if not model_path.exists():
        return None, "Model chưa được train. Vui lòng chạy notebook main.ipynb trước."
    if not preprocessor_path.exists():
        return None, "Preprocessor chưa được tạo. Vui lòng chạy notebook main.ipynb trước."
    
    try:
        model = joblib.load(model_path)
        preprocessor = DataPreprocessor.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        return None, f"Lỗi khi load model: {str(e)}"

model, preprocessor_or_error = load_model()

if model is None:
    st.error(preprocessor_or_error)
    st.info("💡 Hướng dẫn: Chạy notebook `src/main.ipynb` để train model trước khi sử dụng app này.")
else:
    preprocessor = preprocessor_or_error
    
    addresses = load_addresses()
    address_options = [""] + addresses if addresses else [""]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cơ bản")
        address = st.selectbox(
            "Địa chỉ",
            options=address_options,
            index=0,
            help="Chọn địa chỉ từ danh sách"
        )
        area = st.number_input("Diện tích (m²)", min_value=3.0, max_value=600.0, value=60.0, step=1.0)
        frontage = st.number_input("Mặt tiền (m)", min_value=0.0, max_value=80.0, value=5.0, step=0.1)
        access_road = st.number_input("Đường vào (m)", min_value=0.0, max_value=90.0, value=10.0, step=0.1)
        
        house_direction = st.selectbox(
            "Hướng nhà",
            ["", "Đông", "Tây", "Nam", "Bắc", "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"]
        )
        
        balcony_direction = st.selectbox(
            "Hướng ban công",
            ["", "Đông", "Tây", "Nam", "Bắc", "Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc"]
        )
    
    with col2:
        st.subheader("Thông tin chi tiết")
        floors = st.number_input("Số tầng", min_value=1, max_value=10, value=2, step=1)
        bedrooms = st.number_input("Số phòng ngủ", min_value=1, max_value=9, value=2, step=1)
        bathrooms = st.number_input("Số phòng tắm", min_value=1, max_value=9, value=2, step=1)
        
        legal_status = st.selectbox(
            "Tình trạng pháp lý",
            ["", "Have certificate", "Sale contract", "Other"]
        )
        
        furniture_state = st.selectbox(
            "Tình trạng nội thất",
            ["", "Full", "Basic", "None"]
        )
    
    if st.button("🔮 Dự đoán Giá", type="primary", use_container_width=True):
        if not address:
            st.warning("Vui lòng nhập địa chỉ")
        else:
            try:
                input_data = {
                    'Address': [address],
                    'Area': [area],
                    'Frontage': [frontage if frontage > 0 else np.nan],
                    'Access Road': [access_road if access_road > 0 else np.nan],
                    'House direction': [house_direction if house_direction else np.nan],
                    'Balcony direction': [balcony_direction if balcony_direction else np.nan],
                    'Floors': [floors if floors > 0 else np.nan],
                    'Bedrooms': [bedrooms if bedrooms > 0 else np.nan],
                    'Bathrooms': [bathrooms if bathrooms > 0 else np.nan],
                    'Legal status': [legal_status if legal_status else np.nan],
                    'Furniture state': [furniture_state if furniture_state else np.nan],
                    'Price': [np.nan]
                }
                
                df_input = pd.DataFrame(input_data)
                
                X, _ = preprocessor.transform(df_input, target_col='Price')
                
                if 'Floors' in X.columns and 'Area' in X.columns:
                    X['Floors_Area'] = X['Floors'] * X['Area']
                    X['Floors_squared'] = X['Floors'] ** 2
                
                prediction = model.predict(X)[0]
                
                st.success("✅ Dự đoán thành công!")
                
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric("Giá dự đoán", f"{prediction:.2f} tỷ VNĐ")
                with col_pred2:
                    st.metric("Giá dự đoán (USD)", f"${prediction * 40_000:,.0f}")
                with col_pred3:
                    st.metric("Giá/m²", f"{prediction * 1000 / area:.0f} triệu/m²")
                
                st.info(f"💡 Lưu ý: Đây chỉ là dự đoán dựa trên mô hình ML. Giá thực tế có thể khác do nhiều yếu tố khác.")
                
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")
                st.exception(e)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 Thông tin Model", "📈 Đánh giá Mô hình"])
    
    with tab1:
        st.subheader("Thông tin Model")
        metadata_path = base_dir / "src" / "models" / "xgboost_model.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.metric("Loại bài toán", metadata.get('task_type', 'N/A'))
                st.metric("Số features", metadata.get('feature_count', 'N/A'))
                st.metric("Số mẫu train", metadata.get('train_samples', 'N/A'))
            with col_meta2:
                st.metric("CV Folds", metadata.get('cv_folds', 'N/A'))
                st.metric("Tuning method", metadata.get('tuning_method', 'N/A'))
                st.metric("Best CV Score", f"{metadata.get('best_cv_score', 0):.4f}")
            
            with st.expander("Xem chi tiết hyperparameters"):
                st.json(metadata.get('best_params', {}))
        else:
            st.info("Metadata chưa có. Chạy training để tạo metadata.")
    
    with tab2:
        st.subheader("Đánh giá Mô hình")
        
        evaluation_dir = base_dir / "src" / "evaluation"
        report_path = base_dir / "report" / "evaluation.md"
        
        if report_path.exists():
            import re
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metrics = {}
            for line in content.split('\n'):
                if '**' in line and ':' in line:
                    match = re.search(r'\*\*(.*?)\*\*:\s*([\d.]+)', line)
                    if match:
                        metrics[match.group(1)] = float(match.group(2))
            
            if metrics:
                st.markdown("### 📊 Metrics đánh giá")
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                
                with col_met1:
                    st.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
                with col_met2:
                    st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
                with col_met3:
                    st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
                with col_met4:
                    r2 = metrics.get('R2', 0)
                    st.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("### 📈 Biểu đồ đánh giá")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            scatter_path = evaluation_dir / "regression_scatter.png"
            if scatter_path.exists():
                st.image(str(scatter_path), caption="So sánh giá trị thực tế và dự đoán")
            
            feature_imp_path = evaluation_dir / "feature_importance.png"
            if feature_imp_path.exists():
                st.image(str(feature_imp_path), caption="Top 15 Features quan trọng nhất")
        
        with col_chart2:
            residuals_path = evaluation_dir / "residuals_plot.png"
            if residuals_path.exists():
                st.image(str(residuals_path), caption="Biểu đồ Residuals")
            
            corr_path = evaluation_dir / "correlation_matrix.png"
            if corr_path.exists():
                st.image(str(corr_path), caption="Correlation Matrix")

