import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Phát hiện Giao dịch Bất thường trong kiểm toán nội bộ",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Thiết lập Custom CSS để tạo giao diện Premium Dark Mode
st.markdown("""
<style>
    /* Nhập font chữ Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Thiết lập font chữ chủ đạo */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Tùy chỉnh màu nền và chữ */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Thiết kế thẻ KPI cao cấp */
    .kpi-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 20px 35px -10px rgba(16, 185, 129, 0.15);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .kpi-desc {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .kpi-desc.red {
        color: #f87171;
    }
    .kpi-desc.green {
        color: #34d399;
    }
    
    /* Tùy chỉnh các khu vực hiển thị tiêu đề và khối */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .title-gradient {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Làm đẹp thanh Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Custom style cho các tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: white !important;
        border-color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Các hàm hỗ trợ tải dữ liệu & huấn luyện (được cache để tăng tốc độ)
@st.cache_data
def load_data(file_path_or_buffer):
    """
    Đọc dữ liệu từ file CSV, chuyển đổi ngày tháng và tạo các cột đặc trưng.
    """
    df = pd.read_csv(file_path_or_buffer)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['giờ giao dịch'] = df['transaction_date'].dt.hour
    
    # Chuẩn hóa cột nhân viên
    if 'is_employee' in df.columns:
        df['co_nhan_vien'] = df['is_employee'].astype(int)
    else:
        df['co_nhan_vien'] = 0
        df['is_employee'] = False
        
    return df

@st.cache_resource
def train_and_predict(df_features, n_estimators, contamination, random_state):
    """
    Chuẩn hóa đặc trưng và chạy mô hình Isolation Forest để phát hiện bất thường.
    """
    # Chuẩn hóa đặc trưng bằng StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # Tạo và huấn luyện mô hình
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    
    # Tính toán điểm quyết định và nhãn bất thường
    scores = iso.decision_function(X_scaled)
    predictions = iso.predict(X_scaled)
    
    return scores, predictions

# 4. Giao diện Sidebar điều hướng và cấu hình
st.sidebar.markdown("<h2 style='text-align: center; color: #10b981;'>🕵️‍♂️ Điều Khiển</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ["Dữ liệu demo (transactions_Q1_demo.csv)", "Tải lên file dữ liệu mới (.csv)"]
)

uploaded_file = None
if data_source == "Tải lên file dữ liệu mới (.csv)":
    uploaded_file = st.sidebar.file_uploader("Chọn tệp giao dịch CSV:", type=["csv"])

# Cấu hình tham số mô hình
st.sidebar.markdown("### ⚙️ Cấu hình mô hình Isolation Forest")
contamination = st.sidebar.slider(
    "Tỷ lệ rủi ro giả định (Contamination):",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
    format="%.3f"
)

n_estimators = st.sidebar.slider(
    "Số lượng cây quyết định (n_estimators):",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

random_state = st.sidebar.number_input(
    "Random State:",
    value=42,
    step=1
)

# Nút huấn luyện lại mô hình
retrain_button = st.sidebar.button("⚙️ Huấn luyện lại mô hình", use_container_width=True)

# 5. Tiền xử lý và tải dữ liệu chính
data_path = "transactions_Q1_demo.csv"
data_loaded = False

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        data_loaded = True
        st.sidebar.success("Tải dữ liệu mới thành công!")
    else:
        df = load_data(data_path)
        data_loaded = True
except Exception as e:
    st.error(f"Không thể đọc tệp dữ liệu. Lỗi chi tiết: {e}")

# 6. Hiển thị Giao diện chính nếu dữ liệu đã tải thành công
if data_loaded:
    # Trình bày tiêu đề chính với Gradient
    st.markdown('<h1 style="margin-bottom:0px;"><span class="title-gradient">Hệ Thống Phát Hiện Giao Dịch Bất Thường</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-top:5px; margin-bottom:30px;">Phát hiện và phân tích hành vi giao dịch lệch chuẩn sử dụng thuật toán học máy Isolation Forest</p>', unsafe_allow_html=True)

    # Tính toán đặc trưng phục vụ mô hình
    X = df[['amount', 'giờ giao dịch', 'co_nhan_vien']]
    
    # Chạy mô hình
    scores, predictions = train_and_predict(X, n_estimators, contamination, random_state)
    
    # Gán kết quả vào DataFrame
    df["anomaly_score"] = scores
    df["is_anomaly"] = (predictions == -1)

    # 99th percentile số tiền phục vụ lý giải giao dịch lớn
    q99 = df['amount'].quantile(0.99)

    # Định nghĩa hàm lý giải nguyên nhân bất thường
    def explain_anomaly(row):
        reasons = []
        # Kiểm tra lý do ngoài giờ hành chính (giờ < 6 hoặc > 18)
        if row['giờ giao dịch'] < 6 or row['giờ giao dịch'] > 18:
            reasons.append("Ngoài giờ hành chính (18h tối - 6h sáng)")
        # Kiểm tra giao dịch lớn
        if row['amount'] > q99:
            reasons.append("Giá trị giao dịch cực lớn (> 99th percentile)")
        # Kiểm tra giao dịch liên quan đến nhân viên nội bộ
        if row['is_employee']:
            reasons.append("Liên quan tài khoản nhân viên nội bộ")
            
        if not reasons:
            reasons.append("Lệch chuẩn hành vi chung (Isolation Forest)")
            
        return " & ".join(reasons)

    # Thêm lý giải cho các giao dịch bất thường
    df['ly_do_bat_thuong'] = df.apply(explain_anomaly, axis=1)

    # Lọc danh sách bất thường
    df_bat_thuong = df[df["is_anomaly"] == True]
    
    # Lọc danh sách khẩn cấp (điểm rủi ro thấp hơn phân vị 25% của tập bất thường)
    q25_score = df_bat_thuong['anomaly_score'].quantile(0.25)
    df_khan_cap = df_bat_thuong[df_bat_thuong['anomaly_score'] < q25_score]

    # Chia ứng dụng thành 2 tab chính: Dashboard tổng quan và Chi tiết bất thường
    tab_dashboard, tab_details = st.tabs(["📊 Dashboard Tổng Quan & EDA", "🕵️‍♂️ Chi Tiết Giao Dịch Bất Thường"])

    # ---------------- TAB 1: DASHBOARD TỔNG QUAN ----------------
    with tab_dashboard:
        # KPI của tập dữ liệu tổng quan
        st.markdown('<h3 style="margin-bottom:15px; color:#3b82f6;">Chỉ số chính của dữ liệu giao dịch</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tổng giao dịch</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-desc">Tổng số bản ghi giao dịch Q1</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tổng giá trị</div>
                <div class="kpi-value">{df['amount'].sum():,.0f} đ</div>
                <div class="kpi-desc">Tổng lượng tiền giao dịch</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            employee_pct = (df['is_employee'].sum() / len(df)) * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Giao dịch nội bộ</div>
                <div class="kpi-value">{employee_pct:.2f}%</div>
                <div class="kpi-desc">Tỷ lệ giao dịch của nhân viên</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Mức 99th Percentile</div>
                <div class="kpi-value">{q99:,.0f} đ</div>
                <div class="kpi-desc">Ngưỡng giao dịch giá trị lớn (1%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 style="margin-bottom:15px; color:#3b82f6;">Khám phá phân phối dữ liệu (EDA)</h3>', unsafe_allow_html=True)
        
        # Biểu đồ phân bổ
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            hour_counts = df['giờ giao dịch'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Giờ', 'Số lượng giao dịch']
            
            fig_hour = px.bar(
                hour_counts, 
                x='Giờ', 
                y='Số lượng giao dịch',
                title="Phân phối số lượng giao dịch theo giờ trong ngày",
                color='Số lượng giao dịch',
                color_continuous_scale=px.colors.sequential.Viridis,
                template="plotly_dark"
            )
            fig_hour.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with chart_col2:
            fig_amount = px.histogram(
                df[df['amount'] < df['amount'].quantile(0.95)], # Giới hạn 95% để biểu đồ trực quan hơn
                x='amount',
                nbins=50,
                title="Phân phối giá trị giao dịch (Giới hạn dưới phân vị 95%)",
                template="plotly_dark",
                color_discrete_sequence=['#10b981']
            )
            fig_amount.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="Số tiền giao dịch (đ)",
                yaxis_title="Tần suất"
            )
            st.plotly_chart(fig_amount, use_container_width=True)

    # ---------------- TAB 2: CHI TIẾT GIAO DỊCH BẤT THƯỜNG ----------------
    with tab_details:
        # KPI kết quả phát hiện bất thường
        st.markdown('<h3 style="margin-bottom:15px; color:#10b981;">Kết quả phân tích mô hình</h3>', unsafe_allow_html=True)
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Giao dịch bất thường</div>
                <div class="kpi-value" style="color: #ef4444;">{len(df_bat_thuong):,}</div>
                <div class="kpi-desc red">⚠️ Phát hiện {contamination*100:.2f}% tổng giao dịch</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cần xử lý khẩn cấp</div>
                <div class="kpi-value" style="color: #f43f5e;">{len(df_khan_cap):,}</div>
                <div class="kpi-desc red">🚨 Phân vị rủi ro ranh giới (25% nguy cơ nhất)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res3:
            normal_count = len(df) - len(df_bat_thuong)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Giao dịch bình thường</div>
                <div class="kpi-value" style="color: #10b981;">{normal_count:,}</div>
                <div class="kpi-desc green">✔️ Hoạt động bình thường an toàn</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Đặc trưng đầu vào</div>
                <div class="kpi-value">3</div>
                <div class="kpi-desc">Amount, Hour, Is Employee</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Biểu đồ phân bố bất thường (Scatter plot)
        st.markdown('<h3 style="margin-bottom:15px; color:#10b981;">Trực quan hóa vùng rủi ro giao dịch</h3>', unsafe_allow_html=True)
        
        # Mẫu ngẫu nhiên để vẽ đồ thị nhanh hơn nếu tập dữ liệu quá lớn, ở đây 50k vẽ nhanh với WebGL của Plotly
        df_plot = df.copy()
        df_plot['Trạng thái'] = df_plot['is_anomaly'].map({True: 'Bất thường', False: 'Bình thường'})
        
        fig_scatter = px.scatter(
            df_plot,
            x='giờ giao dịch',
            y='amount',
            color='Trạng thái',
            color_discrete_map={'Bình thường': 'rgba(16, 185, 129, 0.2)', 'Bất thường': '#ef4444'},
            title="Biểu đồ phân tán giao dịch theo Giờ và Số tiền",
            labels={'giờ giao dịch': 'Giờ giao dịch trong ngày', 'amount': 'Số tiền giao dịch (đ)'},
            log_y=True, # Sử dụng Log-scale để thấy rõ cả giao dịch nhỏ và lớn
            template="plotly_dark",
            hover_data=['transaction_id', 'transaction_type', 'channel', 'location']
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 style="margin-bottom:15px; color:#10b981;">Danh sách chi tiết & Bộ lọc</h3>', unsafe_allow_html=True)
        
        # Giao diện lọc dữ liệu động
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            locations = df['location'].dropna().unique()
            selected_locations = st.multiselect("Lọc theo Chi nhánh:", options=locations, default=[])
            
        with filter_col2:
            tx_types = df['transaction_type'].dropna().unique()
            selected_tx_types = st.multiselect("Lọc theo Loại giao dịch:", options=tx_types, default=[])
            
        with filter_col3:
            channels = df['channel'].dropna().unique()
            selected_channels = st.multiselect("Lọc theo Kênh thanh toán:", options=channels, default=[])

        # Chia bộ lọc danh sách dựa trên tabs
        subtab_all, subtab_urgent = st.tabs(["⚠️ Tất cả giao dịch bất thường", "🚨 Giao dịch khẩn cấp cần xử lý"])

        # Hàm áp dụng bộ lọc
        def apply_filters(target_df):
            filtered = target_df.copy()
            if selected_locations:
                filtered = filtered[filtered['location'].isin(selected_locations)]
            if selected_tx_types:
                filtered = filtered[filtered['transaction_type'].isin(selected_tx_types)]
            if selected_channels:
                filtered = filtered[filtered['channel'].isin(selected_channels)]
            return filtered

        with subtab_all:
            filtered_all = apply_filters(df_bat_thuong)
            
            # Hiển thị bảng
            display_cols = ['transaction_id', 'transaction_date', 'customer_id_hash', 'amount', 
                            'transaction_type', 'channel', 'location', 'is_employee', 'giờ giao dịch', 
                            'anomaly_score', 'ly_do_bat_thuong']
            
            st.dataframe(
                filtered_all[display_cols].sort_values('anomaly_score'),
                use_container_width=True,
                column_config={
                    "amount": st.column_config.NumberColumn("Số tiền", format="%d đ"),
                    "anomaly_score": st.column_config.NumberColumn("Điểm rủi ro (Càng thấp càng rủi ro)", format="%.4f"),
                    "transaction_date": st.column_config.DatetimeColumn("Ngày giao dịch", format="DD/MM/YYYY HH:mm")
                }
            )
            
            # Xuất dữ liệu
            col_dl1, col_dl2 = st.columns([1, 4])
            with col_dl1:
                # Xuất CSV
                csv_buffer = io.StringIO()
                filtered_all[display_cols].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Tải báo cáo CSV",
                    data=csv_buffer.getvalue(),
                    file_name="tat_ca_bat_thuong.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_dl2:
                # Xuất Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_all[display_cols].to_excel(writer, index=False, sheet_name='Anomalies')
                st.download_button(
                    label="📥 Tải báo cáo Excel (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="tat_ca_bat_thuong.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with subtab_urgent:
            filtered_urgent = apply_filters(df_khan_cap)
            
            st.dataframe(
                filtered_urgent[display_cols].sort_values('anomaly_score'),
                use_container_width=True,
                column_config={
                    "amount": st.column_config.NumberColumn("Số tiền", format="%d đ"),
                    "anomaly_score": st.column_config.NumberColumn("Điểm rủi ro", format="%.4f"),
                    "transaction_date": st.column_config.DatetimeColumn("Ngày giao dịch", format="DD/MM/YYYY HH:mm")
                }
            )
            
            # Xuất dữ liệu khẩn cấp
            col_dl3, col_dl4 = st.columns([1, 4])
            with col_dl3:
                # Xuất CSV
                csv_buffer_urg = io.StringIO()
                filtered_urgent[display_cols].to_csv(csv_buffer_urg, index=False)
                st.download_button(
                    label="📥 Tải báo cáo CSV (Khẩn cấp)",
                    data=csv_buffer_urg.getvalue(),
                    file_name="khan_cap.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_dl4:
                # Xuất Excel
                excel_buffer_urg = io.BytesIO()
                with pd.ExcelWriter(excel_buffer_urg, engine='openpyxl') as writer:
                    filtered_urgent[display_cols].to_excel(writer, index=False, sheet_name='Urgent Anomalies')
                st.download_button(
                    label="📥 Tải báo cáo Excel (Khẩn cấp)",
                    data=excel_buffer_urg.getvalue(),
                    file_name="khan_cap.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
else:
    st.info("💡 Vui lòng thiết lập cấu hình hoặc tải lên file dữ liệu ở thanh Sidebar để bắt đầu phân tích.")
