import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# Đường dẫn đến file Excel
file_path = '\\Nhóm 8 - Thứ 7\\Source Code và Excel\\KNN\\KNN_Dataset_Refined.xlsx'

# Đọc dữ liệu từ file Excel
excel_data = pd.ExcelFile(file_path)

# Đọc các sheet "Training", "Scoring", và "Data" thành DataFrame
training_df = excel_data.parse('Training')
Scoring_df = excel_data.parse('Scoring')
data_df = excel_data.parse('Data')

# Hàm chuẩn hóa dữ liệu
def normalize(column):
    min_val = column.min()
    max_val = column.max()
    return 100 / (max_val - min_val) * (column - min_val)

# Chuẩn hóa dữ liệu trong Training và Data DataFrame
training_df['Normalized Age'] = normalize(training_df['Age'])
training_df['Normalized Income'] = normalize(training_df['Income (1000s)'])
training_df['Normalized Cards'] = normalize(training_df['Cards'])

data_df['Normalized Age'] = normalize(data_df['Age'])
data_df['Normalized Income'] = normalize(data_df['Income (1000s)'])
data_df['Normalized Cards'] = normalize(data_df['Cards'])

# Hàm tính khoảng cách Euclidean
def calculate_normalized_distance(row, data_point):
    return np.sqrt(
        (row['Normalized Age'] - data_point['Normalized Age'])**2 +
        (row['Normalized Income'] - data_point['Normalized Income'])**2 +
        (row['Normalized Cards'] - data_point['Normalized Cards'])**2
    )

# Cột để lưu kết quả dự đoán
data_df['Nhan'] = ''

# Dự đoán cho từng dòng dữ liệu trong DataFrame
for data_index, data_point in data_df.iterrows():
    # Tính khoảng cách từ data_point đến tất cả các điểm trong training_df
    training_df['distance'] = training_df.apply(
        lambda row: calculate_normalized_distance(row, data_point), axis=1
    )
    
    # Lưu các nhãn được dự đoán từ tất cả các giá trị K
    votes = []
    for idx, scoring_row in Scoring_df.iterrows():
        k = int(scoring_row['K'])  # Đảm bảo K là số nguyên
        
        # Lấy K điểm gần nhất
        nearest_neighbors = training_df.nsmallest(k, 'distance')
        
        # Ghi lại nhãn của K điểm lân cận
        votes += list(nearest_neighbors['Response'])
    
    # Xác định nhãn có số lượng lớn nhất
    predicted_label = max(set(votes), key=votes.count)
    data_df.at[data_index, 'Nhan'] = predicted_label

# Tính độ chính xác dựa trên cột Response (Thực Tế)
correct_predictions = (data_df['Nhan'] == data_df['Response (Thực Tế)']).sum()
total_predictions = len(data_df)
accuracy = correct_predictions / total_predictions

# Hiển thị kết quả và độ chính xác
print(data_df[['Age', 'Income (1000s)', 'Cards', 'Nhan', 'Response (Thực Tế)']])
print(f"Độ chính xác của mô hình: {accuracy:.2%}")