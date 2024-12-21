import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Đường dẫn đến file Excel
file_path = '\\Nhóm 8 - Thứ 7\\Source Code và Excel\\KNN\\KNN_Dataset_Refined.xlsx'

# Đọc dữ liệu từ file Excel
excel_data = pd.ExcelFile(file_path)

# Đọc các sheet "Training", "Scoring", và "Data" thành DataFrame
training_df = excel_data.parse('Training')
Scoring_df = excel_data.parse('Scoring')
data_df = excel_data.parse('Data')

# Chuẩn hóa dữ liệu sử dụng MinMaxScaler từ scikit-learn
scaler = MinMaxScaler(feature_range=(0, 100))
training_df[['Normalized Age', 'Normalized Income', 'Normalized Cards']] = scaler.fit_transform(
    training_df[['Age', 'Income (1000s)', 'Cards']]
)
data_df[['Normalized Age', 'Normalized Income', 'Normalized Cards']] = scaler.transform(
    data_df[['Age', 'Income (1000s)', 'Cards']]
)

# Chuẩn bị dữ liệu huấn luyện
X_train = training_df[['Normalized Age', 'Normalized Income', 'Normalized Cards']]
y_train = training_df['Response']

# Tạo cột để lưu kết quả dự đoán
data_df['Nhan'] = ''

# Dự đoán cho từng dòng dữ liệu trong data_df với từng giá trị K
all_predictions = []  # Lưu tất cả các kết quả dự đoán từ các giá trị K

for idx, scoring_row in Scoring_df.iterrows():
    k = int(scoring_row['K'])  # Đảm bảo K là số nguyên
    
    # Tạo mô hình KNN
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    # Lấy dữ liệu cần dự đoán
    X_data = data_df[['Normalized Age', 'Normalized Income', 'Normalized Cards']]
    
    # Dự đoán nhãn cho toàn bộ điểm dữ liệu trong data_df
    predictions = knn.predict(X_data)
    
    # Lưu kết quả dự đoán
    all_predictions.append(predictions)

# Xác định nhãn cuối cùng dựa trên tần suất xuất hiện
for i in range(len(data_df)):
    # Lấy tất cả các dự đoán cho dòng i
    row_predictions = [pred[i] for pred in all_predictions]
    
    # Đếm tần suất xuất hiện của từng nhãn
    most_common_label = Counter(row_predictions).most_common(1)[0][0]
    
    # Gán nhãn xuất hiện nhiều nhất vào cột 'Nhan'
    data_df.at[i, 'Nhan'] = most_common_label

# Tính độ chính xác
correct_predictions = (data_df['Nhan'] == data_df['Response (Thực Tế)']).sum()
total_predictions = len(data_df)
accuracy = correct_predictions / total_predictions

# Hiển thị kết quả và độ chính xác
print(data_df[['Age', 'Income (1000s)', 'Cards', 'Nhan', 'Response (Thực Tế)']])
print(f"Độ chính xác của mô hình: {accuracy:.2%}")

