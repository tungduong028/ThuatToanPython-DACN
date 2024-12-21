from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn tới file Excel
file_path = "\\Nhóm 8 - Thứ 7\\Source Code và Excel\\NaiveBayes\\Lung_Cancer_Dataset_Refined.xlsx"

# Đọc dữ liệu từ các sheet trong file Excel
df_train = pd.read_excel(file_path, sheet_name="Training Data")
df_scoring2 = pd.read_excel(file_path, sheet_name="Scoring Data")
# ====== Phương pháp sử dụng thư viện ======
# Chuẩn bị dữ liệu cho mô hình Naive Bayes của scikit-learn
X_train = df_train.drop(columns=['Lung Cancer'])
y_train = df_train['Lung Cancer']
X_scoring = df_scoring2.drop(columns=['Lung Cancer', 'Lung Cancer (Thực Tế)'])
# Mã hóa các cột dạng chuỗi thành dạng số
label_encoders = {}
X_train_encoded = X_train.copy()
X_scoring_encoded = X_scoring.copy()

for column in X_train_encoded.columns:
    if X_train_encoded[column].dtype == 'object':  # Nếu cột là dạng chuỗi
        le = LabelEncoder()
        X_train_encoded[column] = le.fit_transform(X_train_encoded[column])
        X_scoring_encoded[column] = le.transform(X_scoring_encoded[column])
        label_encoders[column] = le

# Sử dụng GaussianNB để huấn luyện mô hình
model = GaussianNB()
model.fit(X_train_encoded, y_train)

# Dự đoán kết quả
du_doan_ket_qua_thu_vien = model.predict(X_scoring_encoded)

# Thêm kết quả dự đoán vào DataFrame
df_scoring2['Lung Cancer'] = du_doan_ket_qua_thu_vien
print("Kết quả dự đoán thư viện: ")
print(df_scoring2)
# Tính độ chính xác của mô hình
so_mau_dung = (df_scoring2['Lung Cancer'] == df_scoring2['Lung Cancer (Thực Tế)']).sum()
tong_so_mau = len(df_scoring2)
do_chinh_xac = so_mau_dung / tong_so_mau

# Hiển thị độ chính xác
print(f"Độ chính xác của mô hình: {do_chinh_xac * 100:.2f}%")