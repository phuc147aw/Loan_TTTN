import sys
import joblib
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QGridLayout, QHBoxLayout)
from PyQt5.QtCore import Qt

class LoanPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Tải mô hình XGBoost đã lưu
        self.model = joblib.load('XGBoost_model_v2.pkl')

        # Thiết lập giao diện
        self.setWindowTitle('Dự đoán trả nợ')
        self.setGeometry(750, 200, 600, 600)

        self.layout = QVBoxLayout()

        # Label kết quả dự đoán
        self.result_label = QLabel("Dự đoán khả năng trả nợ")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: lightblue; border-radius: 10px; padding: 10px; font-size: 16px;")
        self.layout.addWidget(self.result_label)

        # Tạo các trường nhập liệu
        self.create_input_fields()

        # Tạo một layout ngang cho nút dự đoán
        button_layout = QHBoxLayout()
        
        # Nút dự đoán
        self.predict_button = QPushButton('Dự đoán')
        self.predict_button.setFixedSize(120, 40)  # Kích thước nút nhỏ hơn
        self.predict_button.setStyleSheet("""
            background-color: #007BFF; 
            color: white; 
            border-radius: 10px; 
            padding: 10px; 
            font-size: 12px;
        """)
        self.predict_button.clicked.connect(self.predict)
        
        # Hiệu ứng nhấp chuột cho nút dự đoán
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.pressed.connect(self.on_button_pressed)
        self.predict_button.released.connect(self.on_button_released)

        # Thêm nút dự đoán vào layout ngang và căn giữa
        button_layout.addStretch()  # Thêm khoảng trống bên trái
        button_layout.addWidget(self.predict_button)
        button_layout.addStretch()  # Thêm khoảng trống bên phải

        # Thêm layout nút vào layout chính
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def create_input_fields(self):
        self.inputs = {}
        fields = [
            "Chính sách tín dụng (1: Được chấp nhận, 0: Không được chấp nhận)",
            "Lãi suất (%)",
            "Khoản trả góp hàng tháng",
            "Thu nhập hàng năm (USD)",
            "Tỷ lệ nợ trên thu nhập (DTI)",
            "Điểm FICO",
            "Số ngày có hạn mức tín dụng",
            "Số dư tín dụng quay vòng",
            "Tỷ lệ sử dụng tín dụng quay vòng (%)",
            "Số lần tra cứu tín dụng trong 6 tháng qua",
            "Số lần trễ hạn trong 2 năm qua",
            "Số lần phá sản công khai",
            "Mục đích vay (vd: 'debt_consolidation', 'credit_card', 'home_improvement')"
        ]

        # Sử dụng QGridLayout để căn chỉnh các ô nhập liệu
        grid_layout = QGridLayout()
        
        for i, field in enumerate(fields):
            label = QLabel(field)
            input_field = QLineEdit()
            input_field.setStyleSheet("border: 1px solid #b0b0b0; border-radius: 10px; padding: 5px;")
            grid_layout.addWidget(label, i, 0)
            grid_layout.addWidget(input_field, i, 1)
            self.inputs[field] = input_field

        self.layout.addLayout(grid_layout)

    def on_button_pressed(self):
        """Thay đổi kiểu dáng nút khi nhấn."""
        self.predict_button.setStyleSheet("""
            background-color: #0056b3;  /* Màu tối hơn khi nhấn */
            color: white; 
            border-radius: 10px; 
            padding: 10px; 
            font-size: 12px;
        """)

    def on_button_released(self):
        """Khôi phục kiểu dáng nút khi nhả ra."""
        self.predict_button.setStyleSheet("""
            background-color: #007BFF; 
            color: white; 
            border-radius: 10px; 
            padding: 10px; 
            font-size: 12px;
        """)

    def predict(self):
        try:
            # Nhập dữ liệu từ các trường
            credit_policy = int(self.inputs["Chính sách tín dụng (1: Được chấp nhận, 0: Không được chấp nhận)"].text())
            int_rate = float(self.inputs["Lãi suất (%)"].text())
            installment = float(self.inputs["Khoản trả góp hàng tháng"].text())
            log_annual_inc = np.log(float(self.inputs["Thu nhập hàng năm (USD)"].text()))
            dti = float(self.inputs["Tỷ lệ nợ trên thu nhập (DTI)"].text())
            fico = int(self.inputs["Điểm FICO"].text())
            days_with_cr_line = float(self.inputs["Số ngày có hạn mức tín dụng"].text())
            revol_bal = float(self.inputs["Số dư tín dụng quay vòng"].text())
            revol_util = float(self.inputs["Tỷ lệ sử dụng tín dụng quay vòng (%)"].text())
            inq_last_6mths = int(self.inputs["Số lần tra cứu tín dụng trong 6 tháng qua"].text())
            delinq_2yrs = int(self.inputs["Số lần trễ hạn trong 2 năm qua"].text())
            pub_rec = int(self.inputs["Số lần phá sản công khai"].text())
            purpose = self.inputs["Mục đích vay (vd: 'debt_consolidation', 'credit_card', 'home_improvement')"].text()

            # Tạo DataFrame với dữ liệu đầu vào
            new_customer_data = pd.DataFrame({
                'credit_policy': [credit_policy],
                'int_rate': [int_rate],
                'installment': [installment],
                'log_annual_inc': [log_annual_inc],
                'dti': [dti],
                'fico': [fico],
                'days_with_cr_line': [days_with_cr_line],
                'revol_bal': [revol_bal],
                'revol_util': [revol_util],
                'inq_last_6mths': [inq_last_6mths],
                'delinq_2yrs': [delinq_2yrs],
                'pub_rec': [pub_rec],
                'purpose': [purpose]
            })

            # Dự đoán bằng mô hình đã train
            prediction = self.model.predict(new_customer_data)
            prediction_proba = self.model.predict_proba(new_customer_data)

            prob_full_payment = prediction_proba[0][0]
            prob_non_payment = prediction_proba[0][1]

            # Điều chỉnh dự đoán dựa trên xác suất
            if prob_full_payment - prob_non_payment > 0.1:  # Nếu xác suất trả nợ đầy đủ lớn hơn nhiều
                result = "Trả nợ đầy đủ"
            elif prob_full_payment - prob_non_payment > 0.01:  # Nếu xác suất trả nợ đầy đủ lớn hơn ít
                result = "Có thể trả nợ đầy đủ"
            else:
                result = "Có thể không trả nợ đầy đủ"

            if prob_non_payment - prob_full_payment > 0.1:  # Nếu xác suất không trả nợ đầy đủ lớn hơn nhiều
                result = "Không trả nợ đầy đủ"
            elif prob_non_payment - prob_full_payment > 0.01:  # Nếu xác suất không trả nợ đầy đủ lớn hơn ít
                result = "Có thể không trả nợ đầy đủ"

            # Hiển thị kết quả
            QMessageBox.information(self, 'Kết quả dự đoán',
                                    f'Dự đoán: {result}\n'
                                    f'Xác suất trả nợ đầy đủ: {prob_full_payment:.2f}\n'
                                    f'Xác suất không trả đầy đủ: {prob_non_payment:.2f}')
        except ValueError:
            QMessageBox.warning(self, 'Lỗi', 'Vui lòng nhập đúng định dạng cho tất cả các trường!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoanPredictionApp()
    window.show()
    sys.exit(app.exec_())
