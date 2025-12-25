import pandas as pd
import joblib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from preprocess.preprocess import DataPreprocessor


def predict(model_path, data_path, preprocessor_path=None, output_path=None):
    model = joblib.load(model_path)
    
    df = pd.read_csv(data_path)
    
    if preprocessor_path:
        preprocessor = DataPreprocessor.load(preprocessor_path)
        X, _ = preprocessor.transform(df)
        
        if 'Floors' in X.columns and 'Area' in X.columns:
            X['Floors_Area'] = X['Floors'] * X['Area']
            X['Floors_squared'] = X['Floors'] ** 2
    else:
        raise ValueError("Preprocessor path is required")
    
    predictions = model.predict(X)
    
    result_df = df.copy()
    result_df['prediction'] = predictions
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return predictions, result_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to input data')
    parser.add_argument('--preprocessor', type=str, required=True, help='Path to preprocessor')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file path')
    
    args = parser.parse_args()
    
    predictions, result_df = predict(
        args.model,
        args.data,
        args.preprocessor,
        args.output
    )
    
    print(f"Generated {len(predictions)} predictions")

