# 🕵️‍♂️ Ứng Dụng Phát Hiện Giao Dịch Bất Thường (Transaction Anomaly Detection)

Ứng dụng web được phát triển bằng **Streamlit** giúp phát hiện và phân tích các giao giao dịch tài chính bất thường từ tệp dữ liệu giao dịch sử dụng mô hình học máy **Isolation Forest** (Rừng Cô lập).

Dự án này được chuyển đổi và nâng cấp từ phân tích ban đầu trong file notebook `phat_hien_bat_thuong.ipynb`.

## ✨ Các Tính Năng Chính
* **Tải lên Dữ liệu Linh hoạt**: Hỗ trợ sử dụng dữ liệu mặc định (`transactions_Q1_demo.csv`) hoặc tải lên file dữ liệu giao dịch mới của bạn (.csv).
* **Cấu hình Mô hình Trực tiếp**: Tùy chỉnh trực tiếp các tham số mô hình Isolation Forest (Tỷ lệ rủi ro/Contamination, Số lượng cây quyết định/Estimators) từ thanh điều khiển (Sidebar).
* **Báo cáo và Phân tích Trực quan**:
  * Các chỉ số đo lường chính (KPIs) về doanh thu, tổng số giao dịch và số lượng giao dịch bất thường.
  * Biểu đồ trực quan hóa Plotly tương tác (số lượng giao dịch theo giờ và biểu đồ phân tán phân biệt giao dịch thường/bất thường).
* **Lý giải Nguyên nhân Bất thường**: Tự động phát hiện và cảnh báo các giao dịch rủi ro phát sinh **ngoài giờ hành chính (giờ giao dịch < 6 hoặc > 18)**.
* **Bộ Lọc Động**: Tìm kiếm và lọc danh sách giao dịch bất thường theo Chi nhánh, Loại giao dịch, Kênh thanh toán và Khoảng số tiền.
* **Phân Loại Mức Độ Khẩn Cấp**: Tách riêng các giao dịch có độ rủi ro cực cao (nằm trong phân vị < 25% của điểm số rủi ro) cần xử lý ngay lập tức.
* **Xuất báo cáo**: Tải xuống kết quả phân tích dưới dạng file Excel hoặc CSV để phục vụ báo cáo.

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Cục Bộ

Để khởi chạy ứng dụng dưới máy cục bộ của bạn, hãy làm theo các bước sau:

### 1. Chuẩn bị
Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị từ `3.9` đến `3.11`).

### 2. Tải mã nguồn và cài đặt thư viện
1. Mở terminal (CMD / PowerShell trên Windows hoặc Terminal trên macOS/Linux).
2. Di chuyển đến thư mục chứa mã nguồn của bạn.
3. Tạo môi trường ảo (Khuyến nghị để tránh xung đột thư viện):
   ```bash
   python -m venv venv
   ```
4. Kích hoạt môi trường ảo:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   * **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```
5. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Khởi chạy ứng dụng
Chạy lệnh sau trong terminal:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt mặc định của bạn tại địa chỉ `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Deploy lên Streamlit Cloud (Miễn phí)

Streamlit cung cấp nền tảng **Streamlit Community Cloud** giúp deploy ứng dụng trực tiếp từ kho lưu trữ GitHub của bạn hoàn toàn miễn phí.

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một kho lưu trữ (Repository) mới trên [GitHub](https://github.com).
2. Đẩy toàn bộ các file sau lên GitHub:
   * `app.py`
   * `requirements.txt`
   * `README.md`
   * `transactions_Q1_demo.csv` (File dữ liệu demo để chạy mặc định)
   *(Lưu ý: Không cần đẩy thư mục môi trường ảo `venv` lên GitHub bằng cách tạo file `.gitignore` chứa dòng `venv/`)*

### Bước 2: Deploy lên Streamlit Cloud
1. Truy cập [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấp vào nút **"New app"**.
3. Cấu hình các thông số:
   * **Repository**: Chọn kho lưu trữ GitHub bạn vừa tạo.
   * **Branch**: Chọn nhánh (thường là `main` hoặc `master`).
   * **Main file path**: Điền `app.py`.
4. Nhấp vào nút **"Deploy!"**.
5. Đợi khoảng 1-2 phút để hệ thống tự động cài đặt các thư viện trong `requirements.txt` và khởi tạo ứng dụng. Đường dẫn ứng dụng web công khai sẽ sẵn sàng để chia sẻ!
