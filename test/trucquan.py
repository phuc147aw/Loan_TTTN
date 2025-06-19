import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'loan_data.csv'  # Thay đường dẫn này bằng đường dẫn đến file của bạn
loan_data = pd.read_csv(file_path)

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Distribution of FICO Scores
plt.figure(figsize=(10, 6))
sns.histplot(loan_data['fico'], bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Distribution of FICO Scores', fontsize=16)
plt.xlabel('FICO Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. Distribution of Loan Purposes
plt.figure(figsize=(12, 6))
sns.countplot(data=loan_data, x='purpose', order=loan_data['purpose'].value_counts().index, palette='viridis')
plt.title('Loan Purpose Distribution', fontsize=16)
plt.xlabel('Purpose', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 3. FICO Scores vs Loan Repayment Status
plt.figure(figsize=(10, 6))
sns.boxplot(data=loan_data, x='not_fully_paid', y='fico', palette='Set2')
plt.title('FICO Scores vs Loan Repayment Status', fontsize=16)
plt.xlabel('Not Fully Paid (1 = Yes, 0 = No)', fontsize=12)
plt.ylabel('FICO Score', fontsize=12)
plt.show()