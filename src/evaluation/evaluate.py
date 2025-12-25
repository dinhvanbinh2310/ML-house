import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def evaluate_regression(y_true, y_pred, output_dir='src/evaluation'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('Biểu đồ so sánh giá trị thực tế và dự đoán')
    plt.savefig(output_dir / 'regression_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Residuals')
    plt.title('Biểu đồ Residuals')
    plt.savefig(output_dir / 'residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def evaluate_classification(y_true, y_pred, output_dir='src/evaluation'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1)
    }
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics, cm


def evaluate_clustering(X, labels, output_dir='src/evaluation'):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    silhouette = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    metrics = {
        'Silhouette_Score': float(silhouette),
        'Davies_Bouldin_Score': float(db_score)
    }
    
    return metrics


def save_evaluation_report(metrics, task_type, output_dir='report'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'evaluation.md'
    
    content = f"""# Báo cáo Đánh giá Mô hình

## Loại bài toán: {task_type}

## Metrics

"""
    
    for metric_name, metric_value in metrics.items():
        content += f"- **{metric_name}**: {metric_value:.4f}\n"
    
    content += "\n## Giải thích Metrics\n\n"
    
    if task_type == 'regression':
        content += """
- **RMSE (Root Mean Squared Error)**: Đo lường độ lệch trung bình của dự đoán so với giá trị thực tế. RMSE càng thấp càng tốt.
- **MAE (Mean Absolute Error)**: Trung bình của giá trị tuyệt đối của sai số. MAE càng thấp càng tốt.
- **MAPE (Mean Absolute Percentage Error)**: Phần trăm sai số trung bình. MAPE càng thấp càng tốt.
- **R2 (R-squared)**: Đo lường mức độ phù hợp của mô hình. R2 càng gần 1 càng tốt (tối đa là 1).
"""
    elif task_type == 'classification':
        content += """
- **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu. Accuracy càng cao càng tốt.
- **Precision**: Tỷ lệ dự đoán dương tính thực sự là dương tính. Precision càng cao càng tốt.
- **Recall**: Tỷ lệ các mẫu dương tính thực sự được dự đoán đúng. Recall càng cao càng tốt.
- **F1 Score**: Trung bình điều hòa của Precision và Recall. F1 càng cao càng tốt.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Evaluation report saved to {report_path}")

