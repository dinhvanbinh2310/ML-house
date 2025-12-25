import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from preprocess.preprocess import DataPreprocessor


def train_xgboost(X_train, y_train, task_type='regression', 
                  tuning_method='randomized', cv=5, n_iter=50):
    
    if task_type == 'regression':
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    else:
        base_model = XGBClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    if tuning_method == 'grid':
        search = GridSearchCV(base_model, param_grid, cv=cv, 
                             scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy',
                             n_jobs=-1, verbose=1)
    else:
        # Giảm n_iter mặc định để tránh quá tải bộ nhớ
        actual_n_iter = min(n_iter, 30)
        search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_iter=actual_n_iter,
                                   scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy',
                                   n_jobs=-1, verbose=1, random_state=42)
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    metadata = {
        'task_type': task_type,
        'best_params': best_params,
        'best_cv_score': float(best_score),
        'tuning_method': tuning_method,
        'cv_folds': cv,
        'feature_count': X_train.shape[1],
        'train_samples': len(X_train)
    }
    
    return best_model, metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to processed training data')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--tuning', type=str, default='randomized', choices=['grid', 'randomized'])
    parser.add_argument('--output', type=str, default='src/models/xgboost_model.pkl')
    parser.add_argument('--preprocessor', type=str, default=None, help='Path to saved preprocessor')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    
    if args.preprocessor:
        preprocessor = DataPreprocessor.load(args.preprocessor)
        X, y = preprocessor.transform(df, target_col=args.target)
    else:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col=args.target)
        preprocessor.save('src/models/preprocessor.pkl')
    
    model, metadata = train_xgboost(X, y, task_type=args.task, tuning_method=args.tuning)
    
    model_path = Path(args.output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

