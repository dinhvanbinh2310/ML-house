import pandas as pd
import numpy as np
from pathlib import Path


def generate_descriptive_stats(df, output_dir='report'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_count': df.isnull().sum().to_dict(),
        'missing_rate': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats['numeric_stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75))
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        stats['categorical_stats'][col] = {
            'unique_count': int(df[col].nunique()),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'frequency': df[col].value_counts().to_dict()
        }
    
    stats_df = pd.DataFrame({
        'Column': df.columns,
        'Type': [str(dtype) for dtype in df.dtypes],
        'Missing_Count': [stats['missing_count'][col] for col in df.columns],
        'Missing_Rate_%': [stats['missing_rate'][col] for col in df.columns]
    })
    
    csv_path = output_dir / 'descriptive_stats.csv'
    stats_df.to_csv(csv_path, index=False)
    
    md_content = f"""# Thống kê mô tả Dataset

## Tổng quan
- Số dòng: {stats['shape'][0]}
- Số cột: {stats['shape'][1]}

## Thông tin cột
{stats_df.to_markdown(index=False)}

## Thống kê số học
"""
    for col, col_stats in stats['numeric_stats'].items():
        md_content += f"""
### {col}
- Min: {col_stats['min']:.2f}
- Max: {col_stats['max']:.2f}
- Mean: {col_stats['mean']:.2f}
- Median: {col_stats['median']:.2f}
- Std: {col_stats['std']:.2f}
- Q25: {col_stats['q25']:.2f}
- Q75: {col_stats['q75']:.2f}
"""
    
    md_content += "\n## Thống kê phân loại\n"
    for col, col_stats in stats['categorical_stats'].items():
        md_content += f"""
### {col}
- Số giá trị duy nhất: {col_stats['unique_count']}
- Giá trị xuất hiện nhiều nhất: {col_stats['most_frequent']}
"""
    
    md_path = output_dir / 'descriptive_stats.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return stats

