import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 0. CẤU HÌNH TRANG ĐẦU TIÊN (MANDATORY)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Dự báo PD - Logistic Regression",
    page_icon="🔮"
)

# ==========================================
# 1. HÀM CACHE NẠP DỮ LIỆU DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để tối ưu hóa bộ nhớ và hashable cho st.cache_data"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# ĐỊNH NGHĨA TẬP BIẾN THEO NOTEBOOK
FEATURES = [
    'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
    'NL1', 'NL2', 'NL3', 'NL4', 
    'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
    'TS1', 'TS2', 'TS3', 'TS4'
]
TARGET = 'PD'

# KHOỞI TẠO SESSION STATE ĐỂ LƯU MÔ HÌNH VÀ KẾT QUẢ
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'yhat_test' not in st.session_state:
    st.session_state.yhat_test = None

# ==========================================
# 2. THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV hoặc Excel)", 
        type=["csv", "xlsx"],
        help="Chọn file chứa biến mục tiêu PD và các biến độc lập từ TC1 đến TS4."
    )
    
    st.markdown("---")
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Logistic Regression")
    
    # Các siêu tham số của Logistic Regression theo cấu trúc mở rộng
    random_state = st.number_input(
        "Random State", 
        value=23, 
        step=1,
        help="Giá trị cố định việc xáo trộn dữ liệu khi chia tập Train/Test (Notebook mặc định: 23)"
    )
    
    test_size = st.slider(
        "Tỷ lệ tập kiểm định (Test Size)", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.2, 
        step=0.05,
        help="Tỷ lệ phân chia dữ liệu cho tập Test (Notebook mặc định: 0.2)"
    )
    
    with st.expander("Tham số nâng cao (Solver & C)"):
        c_value = st.number_input("C (Inverse of regularization strength)", value=1.0, min_value=0.01, step=0.1)
        max_iter = st.number_input("Max Iterations", value=100, min_value=50, step=50)
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "saga"])

    st.markdown("---")
    # Nút hành động duy nhất để kích hoạt huấn luyện mô hình
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# 3. THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🔮 Ứng dụng Dự báo Xác suất & Phân lớp (PD)")
st.caption("Hệ thống chuyển đổi tự động từ Notebook phân tích dữ liệu sang ứng dụng web tương tác Streamlit.")

if uploaded_file is None:
    st.info("👋 Chào mừng! Vui lòng tải file dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu khi đã tải file thành công
df = load_data(uploaded_file.getvalue(), uploaded_file.name)

if df is None:
    st.stop()

# Kiểm tra schema dữ liệu bắt buộc
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    st.error(f"🚨 Tệp dữ liệu thiếu các cột bắt buộc sau: {missing_cols}")
    st.stop()

st.caption(f"✅ Đang sử dụng tệp: **{uploaded_file.name}**")
st.markdown("---")

# ZONING MAIN CONTENT INTO TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ==========================================
# 4. THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ==========================================
with tab1:
    st.subheader("📋 Phân tích kích thước & Thống kê mô tả")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng dòng (Bản ghi)", f"{df.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số cột", f"{df.shape[1]}")
    with col_m3:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng File", f"{file_size_mb:.2f} MB")
        
    st.write("#### 🔍 Xem trước 5 dòng dữ liệu đầu tiên (Head)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("#### 📐 Thống kê mô tả các biến trong mô hình")
    # Chỉ thống kê các biến đưa vào mô hình theo quy tắc thiết kế
    st.dataframe(df[FEATURES + [TARGET]].describe().T, use_container_width=True)

# ==========================================
# 5. THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ==========================================
with tab2:
    st.subheader("📊 Phân phối biến mục tiêu & Biến đầu vào độc lập")
    
    # 1. Vẽ phân phối của biến mục tiêu y (Ưu tiên số 1)
    fig_target = px.histogram(
        df, x=TARGET, 
        title=f"Phân phối của biến mục tiêu: {TARGET}",
        color=TARGET,
        barmode="group",
        height=300
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    st.write("#### 🎛️ Biểu đồ phân phối các nhóm biến đầu vào")
    
    # Sử dụng multiselect nếu có quá nhiều biến (24 biến) để tránh tràn màn hình
    default_features = ['TC1', 'NL1', 'DK1', 'TS1']
    selected_features = st.multiselect(
        "Chọn tối đa các biến bạn muốn hiển thị biểu đồ phân phối:",
        options=FEATURES,
        default=default_features,
        max_selections=6
    )
    
    if selected_features:
        # Bố trí lưới biểu đồ cân đối tùy biến theo danh sách chọn
        cols_vis = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with cols_vis[idx % 2]:
                # Tự động chọn biểu đồ dựa trên số lượng giá trị duy nhất (đặc thù dữ liệu phân loại hoặc liên tục)
                if df[col_name].nunique() <= 10:
                    fig_feat = px.bar(
                        df[col_name].value_counts().reset_index(), 
                        x='index', y='count',
                        labels={'index': col_name, 'count': 'Số lượng'},
                        title=f"Tần suất giá trị của biến {col_name}",
                        height=280
                    )
                else:
                    fig_feat = px.histogram(
                        df, x=col_name, 
                        title=f"Phân phối tần suất biến {col_name}",
                        height=280
                    )
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến để hiển thị biểu đồ.")

# ==========================================
# KHỐI XỬ LÝ HUẤN LUYỆN (LƯU TRẠNG THÁI)
# ==========================================
if train_button:
    X = df[FEATURES]
    y = df[TARGET]
    
    # Chia tập train/test theo tham số giao diện
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Khởi tạo và huấn luyện
    model = LogisticRegression(C=c_value, max_iter=max_iter, solver=solver, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Dự báo kiểm định
    yhat_test = model.predict(X_test)
    
    # Tính toán các chỉ tiêu kiểm định phù hợp với phân loại (Classification)
    acc = accuracy_score(y_test, yhat_test)
    cm = confusion_matrix(y_test, yhat_test)
    report_dict = classification_report(y_test, yhat_test, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Lưu toàn bộ vào session_state để không bị mất khi chuyển Tab
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.yhat_test = yhat_test
    st.session_state.metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'report': report_df
    }
    st.sidebar.success("🎉 Huấn luyện mô hình thành công!")

# ==========================================
# 6. THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ==========================================
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng mô hình trên tập Test")
    
    # Điều phối trạng thái rỗng
    if st.session_state.model is None:
        st.info("💡 Vui lòng bấm nút **🚀 Huấn luyện mô hình** tại thanh Sidebar bên trái để xem kết quả kiểm định.")
    else:
        metrics = st.session_state.metrics
        
        # Hiển thị chỉ tiêu vô hướng bằng st.metric
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Độ chính xác toàn cục (Accuracy Score)", f"{metrics['accuracy']:.4f}")
        with col_r2:
            st.metric("Tổng số mẫu kiểm định (Test Size)", f"{len(st.session_state.y_test)} mẫu")
            
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("#### 🧩 Ma trận nhầm lẫn (Confusion Matrix)")
            cm_data = metrics['confusion_matrix']
            # Trực quan hóa trực quan ma trận nhầm lẫn bằng Plotly Heatmap
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán (Predicted)", y="Nhãn Thực Tế (Actual)"),
                x=[f"Lớp {i}" for i in range(cm_data.shape[1])],
                y=[f"Lớp {i}" for i in range(cm_data.shape[0])],
                color_continuous_scale="Blues",
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_c2:
            st.write("#### 📊 Báo cáo phân lớp chi tiết (Classification Report)")
            st.dataframe(metrics['report'].style.format(precision=4), use_container_width=True)

# ==========================================
# 7. THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ==========================================
with tab4:
    st.subheader("🔮 Dự báo đối tượng mới")
    
    # Điều phối trạng thái rỗng
    if st.session_state.model is None:
        st.info("💡 Vui lòng bấm nút **🚀 Huấn luyện mô hình** tại thanh Sidebar bên trái trước khi thực hiện dự báo.")
    else:
        predict_mode = st.radio(
            "Chọn chế độ nhập liệu dự báo:",
            ["Nhập trực tiếp thông số độc lập", "Tải file dữ liệu danh sách mới (Bulk Predict)"],
            horizontal=True
        )
        
        # --- CHẾ ĐỘ 1: NHẬP TRỰC TIẾP ---
        if predict_mode == "Nhập trực tiếp thông số độc lập":
            st.write("✍️ *Thay đổi giá trị của các biến dưới đây để nhận kết quả dự báo thời gian thực:*")
            
            # Sử dụng form để gom các widget, tránh hiện tượng tự động tải lại màn hình liên tục khi chỉnh số
            with st.form("single_prediction_form"):
                
                # Chia lưới các biến để nhập liệu gọn gàng (4 cột)
                form_cols = st.columns(4)
                input_values = []
                
                for idx, feature_name in enumerate(FEATURES):
                    col_target = form_cols[idx % 4]
                    with col_target:
                        # Gợi ý giá trị mặc định là trung vị (hoặc giá trị 4 như mẫu notebook)
                        val = st.number_input(
                            f"Biến {feature_name}", 
                            min_value=0, max_value=10, 
                            value=4, 
                            step=1
                        )
                        input_values.append(val)
                
                submit_pred = st.form_submit_button("🎯 Thực hiện dự báo", type="primary")
                
            if submit_pred:
                X_new = [input_values]
                
                # Gọi hàm predict và predict_proba từ mô hình đã lưu
                prediction = st.session_state.model.predict(X_new)[0]
                probabilities = st.session_state.model.predict_proba(X_new)[0]
                
                st.markdown("---")
                st.write("### Kết quả dự báo:")
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.success(f"**Lớp dự báo (Class): {prediction}**")
                with c_p2:
                    st.info(f"**Xác suất (Probability):** Lớp 0: `{probabilities[0]:.4f}` | Lớp 1: `{probabilities[1]:.4f}`")
                    
        # --- CHẾ ĐỘ 2: DỰ BÁO HÀNG LOẠT QUA FILE ---
        else:
            st.write("📂 *Tải lên file dữ liệu có cấu trúc chứa đúng 24 cột biến độc lập (Từ TC1 đến TS4)*")
            bulk_file = st.file_uploader("Chọn file dữ liệu mới cần chấm điểm", type=["csv", "xlsx"], key="bulk_predict")
            
            if bulk_file is not None:
                df_bulk = load_data(bulk_file.getvalue(), bulk_file.name)
                
                if df_bulk is not None:
                    # Kiểm tra tính khớp của các cột đầu vào
                    missing_bulk_cols = [c for c in FEATURES if c not in df_bulk.columns]
                    if missing_bulk_cols:
                        st.error(f"🚨 Tệp tải lên thiếu các cột đầu vào sau: {missing_bulk_cols}")
                    else:
                        X_bulk = df_bulk[FEATURES]
                        
                        # Dự đoán hàng loạt
                        bulk_preds = st.session_state.model.predict(X_bulk)
                        bulk_probs = st.session_state.model.predict_proba(X_bulk)[:, 1] # Xác suất thuộc lớp 1
                        
                        # Đính kèm cột kết quả vào DataFrame mới
                        df_result = df_bulk.copy()
                        df_result['Dự báo (Predicted_Class)'] = bulk_preds
                        df_result['Xác suất Lớp 1 (Probability_Class_1)'] = bulk_probs
                        
                        st.write("#### 📑 Bảng kết quả dự báo hàng loạt")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Xuất file kết quả dự báo ra CSV
                        csv_output = df_result.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dạng CSV",
                            data=csv_output,
                            file_name="ket_qua_du_bao_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
