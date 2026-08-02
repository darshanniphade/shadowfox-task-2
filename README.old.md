# shadowfox-task-2
# Loan Approval Prediction with Machine Learning

## 📌 Project Overview
Loan Approval Prediction is an important application in the fintech sector. This project uses **Machine Learning** to analyze credit history and applicant data to predict whether a loan application should be approved or not.  
The system considers factors such as:
- Applicant’s financial history
- Income
- Credit rating
- Employment status
- Other relevant attributes

By training a model with historical data, we can make informed predictions for new loan applications.

---

## 📂 Dataset
The dataset contains information about loan applicants, including:
- **Gender**
- **Married**
- **Dependents**
- **Education**
- **Self_Employed**
- **ApplicantIncome**
- **CoapplicantIncome**
- **LoanAmount**
- **Loan_Amount_Term**
- **Credit_History**
- **Property_Area**
- **Loan_Status** (Target: Yes/No)

📌 **Note:** The dataset file (e.g., `loan_data.csv`) should be placed in the project directory.

---

## ⚙️ Technologies Used
- **Python** 🐍
- **Pandas** (Data Handling)
- **NumPy** (Numerical Computations)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-learn** (ML Model Building & Evaluation)
- **Pickle** (Model Saving & Loading)

---

## 📊 Machine Learning Model
- Algorithm Used: **Logistic Regression**
- Model saves as: `loan_model.pkl`
- Preprocessing:
  - Handling missing values
  - Encoding categorical variables
  - Scaling numerical features

---

Enter Gender (Male/Female): Male
Enter Married (Yes/No): Yes
Enter Dependents (0,1,2,3+): 0
Enter Education (Graduate/Not Graduate): Graduate
Enter Self_Employed (Yes/No): No
Enter Applicant Income: 5000
Enter Coapplicant Income: 2000
Enter Loan Amount: 150
Enter Loan Amount Term (in days): 360
Enter Credit History (1/0): 1
Enter Property Area (Urban/Semiurban/Rural): Urban

Prediction: ✅ Loan Approved
