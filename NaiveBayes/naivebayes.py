import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

# Đường dẫn tới file Excel
file_path = "\\Nhóm 8 - Thứ 7\\Source Code và Excel\\NaiveBayes\\Lung_Cancer_Dataset_Refined.xlsx"

# Đọc dữ liệu từ các sheet trong file Excel
df_train = pd.read_excel(file_path, sheet_name="Training Data")
df_scoring = pd.read_excel(file_path, sheet_name="Scoring Data")

# Phân chia tập dữ liệu theo giá trị của cột 'Lung Cancer'
def chia_theo_nhan(df, nhan):
    return df[df['Lung Cancer'] == nhan]

# Tính xác suất P(c) với sửa lỗi Laplace
def tinh_xac_suat_nhan(df, nhan):
    n_classes = len(df['Lung Cancer'].unique())  # Số lượng nhãn phân lớp
    class_count = len(chia_theo_nhan(df, nhan))
    return (class_count + 1) / (len(df) + n_classes)

# Tính xác suất P(x|c) với sửa lỗi Laplace cho từng thuộc tính
def tinh_xac_suat(df, cot, gia_tri):
    unique_values = df[cot].nunique()  # Số lượng giá trị khác nhau của thuộc tính
    count_value = len(df[df[cot] == gia_tri])
    return (count_value + 1) / (len(df) + unique_values)

# Dự đoán nhãn cho từng mẫu trong tập dữ liệu scoring
def du_doan_mau(mau, df_train):
    cac_nhan = df_train['Lung Cancer'].unique()  # Các nhãn có thể có
    xac_suat_hau_nghiem = {}

    # Tính xác suất cho từng nhãn
    for nhan in cac_nhan:
        xac_suat_nhan = tinh_xac_suat_nhan(df_train, nhan)  # P(c)
        xac_suat_cac_thuoc_tinh = 1  # Khởi tạo tích xác suất P(x|c)

        # Tính tích xác suất cho tất cả thuộc tính của mẫu
        for cot in df_train.columns[:-1]:  # Loại bỏ cột 'Lung Cancer'
            xac_suat = tinh_xac_suat(chia_theo_nhan(df_train, nhan), cot, mau[cot])
            xac_suat_cac_thuoc_tinh *= xac_suat
            
        # P(c|x) = P(c) * P(x|c)
        xac_suat_hau_nghiem[nhan] = xac_suat_nhan * xac_suat_cac_thuoc_tinh

    # Trả về nhãn có xác suất lớn nhất
    return max(xac_suat_hau_nghiem, key=xac_suat_hau_nghiem.get)

# Thực hiện dự đoán cho tất cả các mẫu trong tập scoring
du_doan_ket_qua = []
for i, mau in df_scoring.iterrows():
    ket_qua = du_doan_mau(mau, df_train)
    du_doan_ket_qua.append(ket_qua)

# Gán kết quả dự đoán vào cột 'Lung Cancer'
df_scoring['Lung Cancer'] = du_doan_ket_qua

# Hiển thị kết quả dự đoán
print("Kết quả dự đoán: ")
print(df_scoring)

# Tính độ chính xác của mô hình
so_mau_dung = (df_scoring['Lung Cancer'] == df_scoring['Lung Cancer (Thực Tế)']).sum()
tong_so_mau = len(df_scoring)
do_chinh_xac = so_mau_dung / tong_so_mau

# Hiển thị độ chính xác
print(f"Độ chính xác của mô hình: {do_chinh_xac * 100:.2f}%")







