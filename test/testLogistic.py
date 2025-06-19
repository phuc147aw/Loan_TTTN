import pandas as pd
import joblib

# Tải mô hình đã lưu
model_file_path = 'D:\DoAnThucTap\logistic_regression_model_v3.pkl'  # Cập nhật đường dẫn đúng đến mô hình
pipeline = joblib.load(model_file_path)
print(f"Đã tải mô hình từ: {model_file_path}")

# Dữ liệu mẫu để kiểm tra (có thể thay bằng dữ liệu của bạn)
# Ví dụ: tạo một DataFrame với cấu trúc tương tự như dữ liệu huấn luyện
new_data = pd.DataFrame({
    'credit_policy': [1],
    'int_rate': [0.1189],
    'installment': [829.1],
    'log_annual_inc': [10.308],
    'dti': [14.21],
    'fico': [737],
    'days_with_cr_line': [5639],
    'revol_bal': [28854],
    'revol_util': [52.1],
    'inq_last_6mths': [0],
    'delinq_2yrs': [0],
    'pub_rec': [0],
    'purpose': ['debt_consolidation']
})

# Dự đoán với dữ liệu mới
y_pred = pipeline.predict(new_data)
print("Dự đoán:", "Không trả đủ nợ" if y_pred[0] == 1 else "Trả đủ nợ")
