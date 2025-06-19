import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for ,jsonify, session
import os
from datetime import datetime

# Load the trained model
model = joblib.load('logistic_regression_model_v3.pkl')

app = Flask(__name__)
app.secret_key = os.urandom(24)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123"
# Define the path to the CSV file to store history
HISTORY_FILE = 'templates/history.csv'

# Function to load history from CSV
def load_history():
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['time'] = pd.to_datetime(df['time']) # Convert time to datetime objects for sorting
        df = df.sort_values('time', ascending=False) # Sort by time, newest first
        return df.to_dict('records')
    except FileNotFoundError:
        return []

# Function to save history to CSV
def save_history(history):
    df = pd.DataFrame(history)
    df.to_csv(HISTORY_FILE, index=False)

# Load history initially
history = load_history()

@app.route('/')
def about():
    return render_template('about.html')

@app.route('/hme', methods=['GET', 'POST'])
def customer_info():
    if request.method == 'POST':
        cccd = request.form.get('cccd')
        full_name = request.form.get('full_name')
        phone_number = request.form.get('phone_number')

        if not cccd or not full_name or not phone_number:
            return render_template('customer_info.html', error="Vui lòng nhập đầy đủ thông tin!")
        check_tt = []
        for ent in history:
            if str(ent.get('cccd', '')) == str(cccd) and str(ent.get('full_name', '')) == str(full_name):
                check_tt.append(ent)
        customer_history = [entry for entry in history if str(entry.get('cccd', '')) == str(cccd)]  # Filter history by CCCD       
       
       
        if customer_history:
            if check_tt:
                return render_template('index.html',cccd=cccd, history=customer_history, reversed=reversed) 
            else:
                return render_template('customer_info.html', error="Tên không khớp với CCCD. Vui lòng thử lại.")
        else:
            return render_template('index.html',cccd=cccd, history=customer_history,reversed=reversed)  # Pass empty history if no transactions found

    return render_template('customer_info.html')


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')


@app.route('/all_transactions', methods=['GET', 'POST'])
def all_transactions():
    if 'logged_in' not in session:  # Kiểm tra xem đã đăng nhập chưa
        return redirect(url_for('login')) # Chuyển hướng đến trang đăng nhập

    all_history = load_history()
    return render_template('all_transactions.html', history=all_history)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True  # Đánh dấu đã đăng nhập
            return redirect(url_for('all_transactions')) # Chuyển hướng đến trang all_transactions
        else:
            return render_template('login.html', error="Tên đăng nhập hoặc mật khẩu không đúng.")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Xóa session đăng nhập
    return redirect(url_for('login')) # Chuyển hướng về trang chủ

# @app.route('/all_transactions', methods=['GET', 'POST'])
# def all_transactions():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
#             session['logged_in'] = True
#             all_history = load_history()
#             return redirect(url_for('all_transactions_page')) # Chuyển hướng đến route mới
#         else:
#             return jsonify({'error': 'Tên đăng nhập hoặc mật khẩu không đúng.'}), 401

#     if 'logged_in' in session:
#         all_history = load_history()
#         return render_template('all_transactions.html', history=all_history)
#     else:
#         return render_template('index.html', show_login_modal=True)


# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     return redirect(url_for('index'))

# @app.route('/all_transactions_page') # Route mới để hiển thị trang all_transactions.html
# def all_transactions_page():
#     if 'logged_in' in session:
#         all_history = load_history()
#         return render_template('all_transactions.html', history=all_history)
#     else:
#         return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cccd = request.form.get('cccd')
        if not cccd:
            return render_template('customer_info.html', error="CCCD is missing. Please go back and enter your CCCD.")

        # Retrieve input values from the form
        credit_policy = int(request.form['credit_policy'])
        int_rate = float(request.form['int_rate'])
        installment = float(request.form['installment'])
        log_annual_inc = np.log(float(request.form['annual_inc']))
        dti = float(request.form['dti'])
        fico = int(request.form['fico'])
        days_with_cr_line = float(request.form['days_with_cr_line'])
        revol_bal = float(request.form['revol_bal'])
        revol_util = float(request.form['revol_util'])
        inq_last_6mths = int(request.form['inq_last_6mths'])
        delinq_2yrs = int(request.form['delinq_2yrs'])
        pub_rec = int(request.form['pub_rec'])
        purpose = request.form['purpose']

        # Prepare the data as a DataFrame
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

        # Make prediction
        prediction = model.predict(new_customer_data)
        prediction_proba = model.predict_proba(new_customer_data)

        prob_full_payment = prediction_proba[0][0]
        prob_non_payment = prediction_proba[0][1]

        # Interpret the prediction
        if prob_full_payment - prob_non_payment > 0.1:
            result = "Trả nợ đầy đủ"
        elif prob_full_payment - prob_non_payment > 0.01:
            result = "Có thể trả nợ đầy đủ"
        else:
            result = "Có thể không trả nợ đầy đủ"

        if prob_non_payment - prob_full_payment > 0.1:
            result = "Không trả nợ đầy đủ"
        elif prob_non_payment - prob_full_payment > 0.01:
            result = "Có thể không trả nợ đầy đủ"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add to history
        new_entry = {
            'time': current_time,
            'full_name': request.form['full_name'],           
            'phone_number': request.form['phone_number'],           
            'cccd': request.form['cccd'],           
            'result': result,           
            'prob_full_payment': f"{prob_full_payment:.2f}",          
            'prob_non_payment': f"{prob_non_payment:.2f}",
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
        }
        history.append(new_entry)

        # Save updated history to CSV
        save_history(history)

        customer_history = [entry for entry in history if str(entry.get('cccd', '')) == str(cccd)]
        # Filter history by CCCD       
        return render_template('index.html', cccd=cccd, result=result, prob_full_payment=prob_full_payment, prob_non_payment=prob_non_payment, history=customer_history,reversed=reversed)


    except (ValueError, KeyError) as e:
        return render_template('index.html', error=f"Error: {e}", cccd=cccd)

if __name__ == '__main__':
    app.run(debug=True)