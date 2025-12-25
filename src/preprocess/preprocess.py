import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class DataPreprocessor:
    def __init__(self, missing_strategy='mean', outlier_method='iqr', 
                 encoding_method='onehot', scaling_method='standard'):
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method
        
        self.imputers = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.original_numeric_cols = None
        self.original_categorical_cols = None
        
    def handle_missing(self, df, is_fit=False, target_col=None):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if is_fit:
            # Lưu tất cả các cột, kể cả target_col (sẽ loại bỏ sau)
            self.original_numeric_cols = list(numeric_cols)
            # Lưu tất cả categorical_cols, kể cả target_col nếu có
            self.original_categorical_cols = list(categorical_cols)
        
        df_processed = df.copy()
        
        if len(numeric_cols) > 0:
            if self.missing_strategy == 'mean':
                strategy = 'mean'
            elif self.missing_strategy == 'median':
                strategy = 'median'
            elif self.missing_strategy == 'interpolation':
                for col in numeric_cols:
                    df_processed[col] = df_processed[col].interpolate(method='linear')
                return df_processed
            else:
                strategy = 'most_frequent'
            
            imputer = SimpleImputer(strategy=strategy)
            # Đảm bảo thứ tự cột ổn định
            numeric_cols_sorted = sorted(numeric_cols)
            df_processed[numeric_cols_sorted] = imputer.fit_transform(df_processed[numeric_cols_sorted])
            self.imputers['numeric'] = imputer
            # Lưu tên cột đã fit
            self.imputers['numeric_cols'] = numeric_cols_sorted
        
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            # Đảm bảo thứ tự cột ổn định
            # Loại bỏ target_col khỏi danh sách cột để impute (nếu có)
            categorical_cols_to_impute = [col for col in categorical_cols if col != target_col] if target_col else categorical_cols
            if len(categorical_cols_to_impute) > 0:
                categorical_cols_sorted = sorted(categorical_cols_to_impute)
                df_processed[categorical_cols_sorted] = imputer.fit_transform(df_processed[categorical_cols_sorted])
                self.imputers['categorical'] = imputer
                # Lưu tên cột đã fit (không bao gồm target_col)
                self.imputers['categorical_cols'] = categorical_cols_sorted
        
        return df_processed
    
    def handle_outliers(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_processed = df.copy()
        
        if self.outlier_method == 'iqr':
            for col in numeric_cols:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif self.outlier_method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / (df_processed[col].std() + 1e-8))
                df_processed.loc[z_scores >= 3, col] = np.nan
        
        return df_processed
    
    def encode_features(self, df, target_col=None, high_cardinality_threshold=50):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if target_col and target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(target_col)
        
        df_encoded = df.copy()
        
        if self.encoding_method == 'onehot':
            # Tách các cột có cardinality cao và thấp
            high_cardinality_cols = []
            low_cardinality_cols = []
            
            for col in categorical_cols:
                n_unique = df_encoded[col].nunique()
                if n_unique > high_cardinality_threshold:
                    high_cardinality_cols.append(col)
                else:
                    low_cardinality_cols.append(col)
            
            # Label encoding cho các cột có cardinality cao
            if len(high_cardinality_cols) > 0:
                for col in high_cardinality_cols:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[f'label_{col}'] = le
            
            # One-hot encoding cho các cột có cardinality thấp
            if len(low_cardinality_cols) > 0:
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df_encoded[low_cardinality_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(low_cardinality_cols))
                df_encoded = df_encoded.drop(columns=low_cardinality_cols)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                self.encoders['onehot'] = encoder
            else:
                self.encoders['onehot'] = None
        
        elif self.encoding_method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        
        elif self.encoding_method == 'ordinal':
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])
            self.encoders['ordinal'] = encoder
        
        return df_encoded
    
    def scale_features(self, X):
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return X
        
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        if self.feature_names is not None:
            return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        return X_scaled
    
    def fit_transform(self, df, target_col=None):
        df_processed = self.handle_missing(df, is_fit=True, target_col=target_col)
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.encode_features(df_processed, target_col)
        
        if target_col:
            X = df_processed.drop(columns=[target_col], errors='ignore')
            y = df[target_col]
        else:
            X = df_processed
            y = None
        
        X_scaled = self.scale_features(X)
        
        return X_scaled, y
    
    def transform(self, df, target_col=None):
        df_processed = df.copy()
        
        # Xác định các cột numeric và categorical dựa trên danh sách đã lưu khi fit
        if self.original_numeric_cols is not None:
            # Chỉ lấy các cột numeric có trong dataframe và trong danh sách đã lưu
            numeric_cols = [col for col in self.original_numeric_cols if col in df_processed.columns]
        else:
            # Fallback: tự động phát hiện nếu chưa có danh sách
            numeric_cols = list(df_processed.select_dtypes(include=[np.number]).columns)
        
        # Xác định các cột categorical dựa trên danh sách đã lưu khi fit
        if self.original_categorical_cols is not None:
            # Chỉ lấy các cột categorical có trong dataframe và trong danh sách đã lưu
            categorical_cols = [col for col in self.original_categorical_cols 
                              if col in df_processed.columns and col != target_col]
        else:
            # Fallback: tự động phát hiện nếu chưa có danh sách
            categorical_cols = list(df_processed.select_dtypes(include=['object', 'category']).columns)
            if target_col and target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        # Xử lý missing values cho numeric columns
        if 'numeric' in self.imputers and 'numeric_cols' in self.imputers:
            # Sử dụng danh sách cột chính xác mà imputer đã được fit
            numeric_cols_fit = self.imputers['numeric_cols']
            numeric_cols_ordered = [col for col in numeric_cols_fit if col in df_processed.columns]
            
            if len(numeric_cols_ordered) > 0:
                # Đảm bảo thứ tự khớp với lúc fit
                df_processed[numeric_cols_ordered] = self.imputers['numeric'].transform(
                    df_processed[numeric_cols_ordered]
                )
        
        # Xử lý missing values cho categorical columns
        if 'categorical' in self.imputers and 'categorical_cols' in self.imputers:
            # Sử dụng danh sách cột chính xác mà imputer đã được fit
            categorical_cols_fit = self.imputers['categorical_cols']
            # CHỈ lấy các cột có trong danh sách đã fit (không lấy cột mới)
            categorical_cols_ordered = [col for col in categorical_cols_fit 
                                      if col in df_processed.columns and col != target_col]
            
            # Phải có đủ số cột như lúc fit (có thể ít hơn nếu thiếu, nhưng không được nhiều hơn)
            if len(categorical_cols_ordered) == len(categorical_cols_fit):
                # Đảm bảo thứ tự khớp với lúc fit
                df_processed[categorical_cols_ordered] = self.imputers['categorical'].transform(
                    df_processed[categorical_cols_ordered]
                )
            elif len(categorical_cols_ordered) < len(categorical_cols_fit):
                # Thiếu một số cột - có thể do cột không có trong input
                # Thêm các cột thiếu với giá trị NaN để đảm bảo thứ tự
                missing_cols = [col for col in categorical_cols_fit if col not in categorical_cols_ordered]
                for col in missing_cols:
                    if col != target_col:
                        df_processed[col] = np.nan
                # Sắp xếp lại theo thứ tự fit
                categorical_cols_ordered = [col for col in categorical_cols_fit 
                                          if col != target_col]
                df_processed[categorical_cols_ordered] = self.imputers['categorical'].transform(
                    df_processed[categorical_cols_ordered]
                )
        
        df_processed = self.handle_outliers(df_processed)
        
        # Sử dụng lại danh sách categorical_cols đã xác định ở trên
        # Chuyển thành Index để tương thích với code sau
        categorical_cols = pd.Index(categorical_cols) if len(categorical_cols) > 0 else pd.Index([])
        
        if self.encoding_method == 'onehot':
            # Xử lý các cột có cardinality cao (đã được label encode)
            high_cardinality_cols = []
            low_cardinality_cols = []
            
            for col in categorical_cols:
                if f'label_{col}' in self.encoders:
                    high_cardinality_cols.append(col)
                else:
                    low_cardinality_cols.append(col)
            
            # Transform label encoding cho các cột có cardinality cao
            for col in high_cardinality_cols:
                if f'label_{col}' in self.encoders:
                    le = self.encoders[f'label_{col}']
                    try:
                        df_processed[col] = le.transform(df_processed[col].astype(str))
                    except ValueError:
                        # Nếu có giá trị mới chưa thấy, gán giá trị mặc định
                        unique_vals = set(le.classes_)
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in unique_vals else -1
                        )
            
            # Transform one-hot encoding cho các cột có cardinality thấp
            if len(low_cardinality_cols) > 0 and self.encoders.get('onehot') is not None:
                encoder = self.encoders['onehot']
                encoded = encoder.transform(df_processed[low_cardinality_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(low_cardinality_cols))
                df_processed = df_processed.drop(columns=low_cardinality_cols)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            elif len(low_cardinality_cols) > 0:
                # Nếu không có encoder, drop các cột này
                df_processed = df_processed.drop(columns=low_cardinality_cols)
        
        elif self.encoding_method == 'label':
            for col in categorical_cols:
                if col in self.encoders:
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
        
        elif self.encoding_method == 'ordinal' and 'ordinal' in self.encoders:
            df_processed[categorical_cols] = self.encoders['ordinal'].transform(df_processed[categorical_cols])
        
        if target_col:
            X = df_processed.drop(columns=[target_col], errors='ignore')
            y = df[target_col] if target_col in df.columns else None
        else:
            X = df_processed
            y = None
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            if self.feature_names is not None:
                X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        else:
            X_scaled = X
        
        return X_scaled, y
    
    def save(self, path):
        joblib.dump({
            'imputers': self.imputers,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'missing_strategy': self.missing_strategy,
            'outlier_method': self.outlier_method,
            'encoding_method': self.encoding_method,
            'scaling_method': self.scaling_method,
            'original_numeric_cols': self.original_numeric_cols,
            'original_categorical_cols': self.original_categorical_cols
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        preprocessor = cls(
            missing_strategy=data['missing_strategy'],
            outlier_method=data['outlier_method'],
            encoding_method=data['encoding_method'],
            scaling_method=data['scaling_method']
        )
        preprocessor.imputers = data['imputers']
        preprocessor.encoders = data['encoders']
        preprocessor.scaler = data['scaler']
        preprocessor.feature_names = data['feature_names']
        preprocessor.original_numeric_cols = data.get('original_numeric_cols')
        preprocessor.original_categorical_cols = data.get('original_categorical_cols')
        return preprocessor

