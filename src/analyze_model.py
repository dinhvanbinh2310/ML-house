"""
Script de phan tich feature importance va correlation
Chay sau khi da train model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load model và data
base_dir = Path(__file__).parent.parent
model_path = base_dir / "src" / "models" / "xgboost_model.pkl"
data_path = base_dir / "data" / "raw" / "vietnam_housing_dataset.csv"
X_train_path = base_dir / "data" / "processed" / "X_train.csv"

# Load model
model = joblib.load(model_path)
print("Da load model")

# Load data
df = pd.read_csv(data_path)
X_train = pd.read_csv(X_train_path)
print("Da load data")

# Phân tích Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 FEATURES QUAN TRỌNG NHẤT ===")
print(feature_importance.head(15).to_string(index=False))

# Vẽ biểu đồ feature importance
output_dir = base_dir / "src" / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nDa luu bieu do feature importance vao {output_dir / 'feature_importance.png'}")
plt.close()

# Kiểm tra correlation giữa các features số và target
numeric_features = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
available_features = [f for f in numeric_features if f in df.columns]

if len(available_features) > 0:
    correlation_data = df[available_features + ['Price']].corr()['Price'].sort_values(ascending=False)
    
    print("\n=== CORRELATION VỚI TARGET (Price) ===")
    print(correlation_data)
    
    # Vẽ correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[available_features + ['Price']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Numeric Features vs Price')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Da luu correlation matrix vao {output_dir / 'correlation_matrix.png'}")
    plt.close()
    
    print("\nLUU Y:")
    print("- Nếu correlation âm: feature tăng thì giá giảm (có thể hợp lý)")
    print("- Nếu correlation dương: feature tăng thì giá tăng (hợp lý)")
    print("- Kiểm tra feature importance để xem model học gì")

# Kiểm tra các features quan trọng liên quan đến Floors, Bedrooms, Bathrooms
important_numeric = ['Floors', 'Bedrooms', 'Bathrooms', 'Area']
for feat in important_numeric:
    if feat in feature_importance['feature'].values:
        idx = feature_importance[feature_importance['feature'] == feat].index[0]
        importance = feature_importance.loc[idx, 'importance']
        print(f"\n{feat}:")
        print(f"  - Feature Importance: {importance:.4f}")
        print(f"  - Rank: {idx + 1}/{len(feature_importance)}")
        if feat in df.columns:
            corr = df[feat].corr(df['Price'])
            print(f"  - Correlation với Price: {corr:.4f}")

