import pandas as pd
import numpy as np
import streamlit as st
import os

# --- Imports cho các mô hình Scikit-learn ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Imports cho các mô hình bên ngoài (CẦN PHẢI CÀI ĐẶT BẰNG PIP) ---
try:
    import xgboost as xgb
    XGBRegressor = xgb.XGBRegressor
except ImportError:
    XGBRegressor = None
    
try:
    import lightgbm as lgb
    LGBMRegressor = lgb.LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    import catboost as cat
    CatBoostRegressor = cat.CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

# -----------------------------------------------------------------------------------
# THAM SỐ CẤU HÌNH VÀ TÊN FILE
# -----------------------------------------------------------------------------------
RAW_DATA_FILE = "Data_tho_chua_xu_ly.csv"
TARGET_COL = 'RON 95-III(VND)'
TEST_SIZE = 150 
LAG_W = [1, 7] 
VOL_W = 7      
EVENT_LAG = [3, 7] 

EVENT_MAP = {
    'Cung (OPEC & Sản lượng)': 'event_Cung (OPEC & Sản lượng)',
    'Cung (Tồn kho Mỹ)': 'event_Cung (Tồn kho Mỹ)',
    'Cầu (Kinh tế vĩ mô)': 'event_Cầu (Kinh tế vĩ mô)',
    'Sự cố & Gián đoạn': 'event_Sự cố & Gián đoạn',
    'Địa chính trị & Xung đột': 'event_Địa chính trị & Xung đột',
    'Đồng USD & Tài chính': 'event_Đồng USD & Tài chính'
}

# -----------------------------------------------------------------------------------
# HÀM FEATURE ENGINEERING VÀ SCALING (Giữ nguyên)
# -----------------------------------------------------------------------------------
def create_features(df_raw, scaler=None, fit_scaler=False):
    """Thực hiện toàn bộ quá trình Feature Engineering và Scaling/Transforming."""
    df = df_raw.copy()
    
    # 1. Basic Cleaning
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    cols_to_fill = ['Gia_Brent(USD)', 'Gia_WTI(USD)', 'USD/VND', 'Bien_loi_nhuan']
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()
    
    df = df.drop(columns=['E5 RON 92-II(VND)', 'Bien_loi_nhuan']) 

    # 2. Time Series Features
    price_cols = ['Gia_Brent(USD)', 'Gia_WTI(USD)', 'USD/VND']
    
    for col in price_cols:
        col_name_base = col.split("(")[0]
        for lag in LAG_W:
            df[f'{col_name_base}_lag{lag}'] = df[col].shift(lag)
        
        df[f'{col_name_base}_pct'] = df[col].pct_change()
        df[f'{col_name_base}_vol{VOL_W}'] = df[col].rolling(window=VOL_W).std()
        
    df = df.dropna()

    # 3. Event Features
    df['loai_su_kien'] = df['loai_su_kien'].fillna('No_Event')
    df['tang_giam'] = df['tang_giam'].fillna('None')

    event_dummies = pd.get_dummies(df['loai_su_kien']).astype(int)
    event_dummies = event_dummies.rename(columns={k: v for k, v in zip(EVENT_MAP.keys(), EVENT_MAP.values()) if k in event_dummies.columns})
    
    for col in EVENT_MAP.values():
        if col not in event_dummies.columns:
            event_dummies[col] = 0
            
    if 'No_Event' in event_dummies.columns:
        event_dummies = event_dummies.drop(columns=['No_Event'])
    
    df['event_impact'] = (df['loai_su_kien'] != 'No_Event').astype(int)

    sentiment_map = {'Giảm': -1, 'Tăng': 1, 'None': 0}
    df['sentiment_score'] = df['tang_giam'].map(sentiment_map)
    df['event_sentiment_7'] = df['sentiment_score'].rolling(window=VOL_W).sum()
    df = df.drop(columns=['sentiment_score']) 

    for lag in EVENT_LAG:
        df[f'event_lag_{lag}'] = df['event_impact'].shift(1).rolling(window=lag).sum()

    df = pd.concat([df.drop(columns=['loai_su_kien', 'tang_giam', 'ten_su_kien']), event_dummies], axis=1).dropna()
    
    y_raw = df[TARGET_COL]
    X_features = df.drop(columns=[TARGET_COL])
    
    # 4. Scaling (Standard Scaling)
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        X_scaled_df = pd.DataFrame(X_scaled, index=X_features.index, columns=X_features.columns)
        return X_scaled_df, y_raw, scaler
    
    elif scaler is not None:
        X_scaled = scaler.transform(X_features)
        X_scaled_df = pd.DataFrame(X_scaled, index=X_features.index, columns=X_features.columns)
        return X_scaled_df, y_raw

    return X_features, y_raw 

# -----------------------------------------------------------------------------------
# HÀM TẢI VÀ HUẤN LUYỆN NHIỀU MÔ HÌNH (ĐÃ CẬP NHẬT)
# -----------------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    """Tải dữ liệu, chuẩn bị, và huấn luyện nhiều mô hình."""
    if not os.path.exists(RAW_DATA_FILE):
        st.error(f"File dataset '{RAW_DATA_FILE}' không tìm thấy.")
        return None, None, None, None, None

    df_raw = pd.read_csv(RAW_DATA_FILE)

    X_scaled, y_raw, scaler = create_features(df_raw, fit_scaler=True)
    
    X_train = X_scaled.iloc[:-TEST_SIZE]
    X_test = X_scaled.iloc[-TEST_SIZE:]
    y_train = y_raw.iloc[:-TEST_SIZE]
    y_test = y_raw.iloc[-TEST_SIZE:]

    # Định nghĩa các mô hình
    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }
    
    # Thêm các mô hình bên ngoài nếu đã cài đặt
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if LGBMRegressor is not None:
        # Tắt verbose cho LightGBM
        models["LightGBM Regressor"] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    if CatBoostRegressor is not None:
        # Tắt hiển thị tiến trình cho CatBoost
        models["CatBoost Regressor"] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)


    # Huấn luyện mô hình và lưu RMSE
    model_results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train.values)
            y_pred_test = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            model_results[name] = {'model': model, 'rmse': rmse}
        except Exception as e:
            # Bỏ qua mô hình nếu có lỗi huấn luyện (ví dụ: thiếu thư viện)
            st.warning(f"Không thể huấn luyện mô hình {name}. Lỗi: {e}")


    # Tìm mô hình tốt nhất để hiển thị mặc định
    best_model_name = min(model_results, key=lambda k: model_results[k]['rmse'])
    
    feature_names = X_scaled.columns.tolist()

    return model_results, best_model_name, feature_names, scaler, df_raw

# -----------------------------------------------------------------------------------
# HÀM DỰ ĐOÁN VỚI INPUT THÔ (Giữ nguyên)
# -----------------------------------------------------------------------------------
def predict_raw_input(raw_input_dict, df_raw_full, feature_names, scaler, selected_model):
    """Thực hiện dự đoán từ input thô của người dùng bằng mô hình đã chọn."""
    
    df_full_history = df_raw_full.copy()
    
    input_row_series = pd.Series({
        'date': pd.to_datetime(raw_input_dict['date']),
        'Gia_Brent(USD)': raw_input_dict['Gia_Brent(USD)'],
        'Gia_WTI(USD)': raw_input_dict['Gia_WTI(USD)'],
        'USD/VND': raw_input_dict['USD/VND'],
        'loai_su_kien': raw_input_dict['loai_su_kien'],
        'ten_su_kien': np.nan, 
        'tang_giam': raw_input_dict['tang_giam'],
        'E5 RON 92-II(VND)': np.nan, 
        'RON 95-III(VND)': np.nan,   
        'Bien_loi_nhuan': np.nan 
    })
    
    df_full_history.loc[len(df_full_history)] = input_row_series
    
    X_full, _ = create_features(df_full_history, scaler=scaler, fit_scaler=False)
    
    X_predict = X_full.iloc[[-1]]
    X_predict = X_predict[feature_names] 
    
    raw_prediction = selected_model.predict(X_predict)[0]
    
    return raw_prediction, X_predict

# -----------------------------------------------------------------------------------
# PHẦN CHÍNH CỦA STREAMLIT APP
# -----------------------------------------------------------------------------------

# Tải và huấn luyện mô hình
# (Các mô hình không cài đặt sẽ trả về lỗi, nhưng ứng dụng vẫn chạy với các mô hình khả dụng)
model_results, best_model_name, feature_names, scaler, df_raw = load_and_train_model()

# --- Kiểm tra nếu mô hình tải thành công ---
if df_raw is None:
    st.stop() 

default_values_raw = df_raw.iloc[-1] 

# ----------------- Giao diện Streamlit -----------------

st.set_page_config(page_title="Dự đoán Giá Xăng RON 95-III", layout="wide")
st.title("⛽ Ứng dụng Dự đoán Giá Xăng RON 95-III Nội địa")

# 1. Hướng dẫn cài đặt
st.markdown("""
    <div style='background-color:#fff3cd; color:#856404; padding: 10px; border-radius: 5px;'>
        <strong>⚠️ HƯỚNG DẪN:</strong> Để sử dụng các mô hình Gradient Boosting nâng cao (XGBoost, LightGBM, CatBoost), 
        bạn phải cài đặt chúng trong môi trường của mình:
        <br><code>pip install xgboost lightgbm catboost</code>
    </div>
    """, unsafe_allow_html=True)


# Bảng so sánh RMSE
st.sidebar.subheader("📊 So sánh Hiệu suất Mô hình (RMSE - VND)")
rmse_data = {
    'Mô hình': list(model_results.keys()),
    'RMSE (VND)': [f"{model_results[name]['rmse']:,.0f}" for name in model_results.keys()]
}
rmse_df = pd.DataFrame(rmse_data)
st.sidebar.dataframe(rmse_df.set_index('Mô hình'), use_container_width=True)

# Lựa chọn mô hình
model_selection = st.sidebar.selectbox(
    "Chọn Mô hình Dự đoán",
    options=list(model_results.keys()),
    index=list(model_results.keys()).index(best_model_name)
)

st.markdown(f"""
    <p style='font-size:18px;'>
    Mô hình đang được sử dụng: <b>{model_selection}</b>. 
    RMSE trên tập kiểm tra: <b>{model_results[model_selection]['rmse']:,.0f} VND</b>.
    </p>
    """, unsafe_allow_html=True)

st.sidebar.header("Thông tin Đầu vào Dự đoán (Giá trị THÔ)")

# Raw Price Inputs
input_prices = {}
price_fields = [
    ('Gia_Brent(USD)', 'Giá Brent (USD)'),
    ('Gia_WTI(USD)', 'Giá WTI (USD)'),
    ('USD/VND', 'Tỷ giá USD/VND')
]

st.sidebar.subheader("I. Giá Hàng hóa & Tỷ giá")
for feature_name, label in price_fields:
    default_val = float(default_values_raw[feature_name]) if not pd.isna(default_values_raw[feature_name]) else 70.0
    input_prices[feature_name] = st.sidebar.number_input(
        label,
        value=default_val,
        step=0.01,
        format="%.2f",
        key=f"raw_input_{feature_name}"
    )

# Event Inputs
st.sidebar.subheader("II. Thông tin Sự kiện")

unique_events = list(EVENT_MAP.keys())
unique_events.insert(0, 'Không có sự kiện')

selected_event = st.sidebar.selectbox(
    "Loại Sự kiện",
    options=unique_events,
    index=0
)

sentiment = st.sidebar.radio(
    "Xu hướng Sự kiện",
    options=['None', 'Tăng', 'Giảm'],
    index=0,
    disabled=(selected_event == 'Không có sự kiện')
)

# Date input 
last_date = pd.to_datetime(df_raw.iloc[-1]['date'])
input_date = st.sidebar.date_input(
    "Ngày Dự đoán",
    value=last_date + pd.Timedelta(days=1),
    min_value=last_date + pd.Timedelta(days=1),
    key="input_date"
)

# ----------------- Nút Dự đoán -----------------

st.sidebar.markdown("---")
if st.sidebar.button("Dự đoán Giá Xăng (VND)", type="primary"):
    selected_model = model_results[model_selection]['model']
    
    if selected_model is not None:
        st.header("Kết quả Dự đoán")
        
        raw_input_data = {
            'date': input_date.strftime('%Y-%m-%d'),
            'Gia_Brent(USD)': input_prices['Gia_Brent(USD)'],
            'Gia_WTI(USD)': input_prices['Gia_WTI(USD)'],
            'USD/VND': input_prices['USD/VND'],
            'loai_su_kien': selected_event if selected_event != 'Không có sự kiện' else np.nan,
            'ten_su_kien': np.nan, 
            'tang_giam': sentiment if sentiment != 'None' else np.nan,
        }
        
        try:
            raw_prediction, X_predict = predict_raw_input(raw_input_data, df_raw, feature_names, scaler, selected_model)
            
            st.success(f"### Dự đoán Giá RON 95-III (Thực tế): **{raw_prediction:,.0f} VND**")
            
            st.markdown("#### Vector Đặc trưng Đã Chuẩn hóa (Scaled Features) được sử dụng:")
            
            X_predict_T = X_predict.T
            X_predict_T.columns = ["Giá trị Đã Chuẩn hóa"]
            st.dataframe(X_predict_T, use_container_width=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")

# Footer
st.markdown("---")
st.markdown("Dashboard được tạo ra để minh họa khả năng dự đoán chuỗi thời gian bằng các mô hình khác nhau.")