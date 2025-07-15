import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Streamlit Page Configuration:
st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",page_icon="💰",layout="wide")

# Dataset Loading:
@st.cache_data
def load_data():
    conn = sqlite3.connect("credit_risk.db")
    df = pd.read_sql_query("SELECT * FROM credit_risk", conn)
    conn.close()
    return df

df=load_data()

# Main content
st.header("Credit Risk Analyzer")
st.sidebar.markdown(""" 
        Welcome to the Credit Risk Analysis Dashboard!
  
        Use this sidebar to navigate! """)

# KPIs:
st.markdown("## 📌Metrics")
col1, = st.columns(1)
total_loan = df["loan_amnt"].sum()
col1.metric("Total Loan Amount", f"{total_loan}")

# --- Sidebar Navigation---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", ["🏠Dashboard Home","📈Dataset Analysis", "🔮 Predict Credit Risk"])

# Load Model:
model = joblib.load("credit_risk_model.pkl")

# Pages:
if selection == "🏠Dashboard Home":
    fig1 = px.scatter(df,x='person_age',y='person_income',title='Age and Income',color_discrete_sequence=['gray'],)
    fig1.update_traces(marker=dict(line=dict(width=1, color='green')))
    fig1.update_layout(xaxis_title='Age',yaxis_title='Income',plot_bgcolor='white',margin=dict(l=40, r=20, t=40, b=40),showlegend=False)
    st.plotly_chart(fig1)

    fig3 = px.scatter(df,x='loan_amnt',y='person_income',title='Loan Amount as per Income',color_discrete_sequence=['gray'],)
    fig3.update_traces(marker=dict(line=dict(width=1, color='blue')))
    fig3.update_layout(xaxis_title='Loan Amount',yaxis_title='Income',plot_bgcolor='white',margin=dict(l=40, r=20, t=40, b=40),showlegend=False)
    st.plotly_chart(fig3)

elif selection == "📈Dataset Analysis":
    st.title("Dataset Overview")
    st.dataframe(df)
    st.write("Summary Statistics:")
    st.write(df.describe())

    df['default_status']=df['loan_status'].apply(lambda x:'Default' if x==1 else 'Non-Default')
    
    plt.figure(figsize=(5, 4))
    defau=df['default_status']
    defau_count=df['default_status'].value_counts()
    ax=sns.countplot(x='default_status', data=df, palette='viridis')
    plt.title('Distribution of Loan Status', fontsize=14)
    plt.xlabel('Loan Status')
    plt.ylabel('Number of Applicants')
    for p in ax.patches:
        height= p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5, int(height), ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    loan_amount_avg=df.groupby('default_status')['loan_amnt'].mean().reset_index()
    defau=df['default_status']
    defau_count=df['default_status'].value_counts()
    plt.figure(figsize=(6, 4))
    al=sns.barplot(x='default_status', y='loan_amnt', data=loan_amount_avg, palette='magma')
    plt.title('Average Loan Amount by Loan Status', fontsize=14)
    plt.xlabel('Loan Status', fontsize=12)
    plt.ylabel('Average Loan Amount ($)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for p in al.patches:
        height= p.get_height()
        al.text(p.get_x() + p.get_width()/2., height + 5, int(height), ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x='loan_percent_income', hue='default_status', fill=True, common_norm=False, palette='coolwarm')
    plt.title('Debt-to-Income Ratio (loan_percent_income) by Loan Status', fontsize=14)
    plt.xlabel('Debt-to-Income Ratio')
    plt.ylabel('Density')
    plt.show()

    st.subheader("Relationship By Columns")
    x_feature = st.selectbox("Select a Column", df.columns, key="x_feature_selectbox")
    y_feature = st.selectbox("Select a Column", df.columns, key="y_feature_selectbox")
    color_col= st.selectbox("Color by", df.columns)
    fig4 = px.scatter(df, x=x_feature, y=y_feature, color=color_col)
    st.plotly_chart(fig4)

elif selection == "🔮 Predict Credit Risk":
    # Mapping dictionaries for categorical inputs
    home_ownership_map = {'RENT': 1, 'MORTGAGE': 2, 'OWN': 3, 'OTHER': 4}
    loan_intent_map = {'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3,
        'PERSONAL': 4, 'DEBTCONSOLIDATION': 5, 'HOMEIMPROVEMENT': 6}
    loan_grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    default_map = {'N': 0, 'Y': 1}

    # Streamlit UI
    st.title("💳 Credit Risk Recommendation System")
    st.markdown("Fill in the details below to predict the credit risk score of an applicant.")

    # User Inputs
    income = st.number_input("Income ($)", min_value=0.0, step=100.0, format="%.2f")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=100.0, format="%.2f")
    credit_length = st.number_input("Credit History Length (years)", min_value=0, step=1)
    loan_percent_income = st.slider("Loan Percent Income (0.0 - 1.0)", min_value=0.0, max_value=1.0, step=0.01)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    emp_length = st.number_input("Employment Length (years)", min_value=0, step=1)

    home_ownership = st.selectbox("Home Ownership", list(home_ownership_map.keys()))
    loan_intent = st.selectbox("Loan Intent", list(loan_intent_map.keys()))
    loan_grade = st.selectbox("Loan Grade", list(loan_grade_map.keys()))
    defaulted_before = st.radio("Previously Defaulted?", list(default_map.keys()))

    # Load Model:
    @st.cache_resource
    def load_model():
        return joblib.load("credit_risk_model.pkl")
    
    model = joblib.load("credit_risk_model.pkl")

    # Prediction Logic
    if st.button("🔍 Predict Risk"):
        try:
            input_data = {
                'person_income': income,
                'loan_amnt': loan_amount,
                'cb_person_cred_hist_length': credit_length,
                'loan_percent_income': loan_percent_income,
                'person_age': age,
                'person_emp_length': emp_length,
                'person_home_ownership': home_ownership_map[home_ownership],
                'loan_intent': loan_intent_map[loan_intent],
                'loan_grade': loan_grade_map[loan_grade],
                'cb_person_default_on_file': default_map[defaulted_before]
            }

            input_df = pd.DataFrame([input_data])

            # Make prediction
            prob_default = model.predict_proba(input_df)[:, 1][0]
            risk_score = (1 - prob_default) * 10

            # Determine classification
            if risk_score > 7:
                risk_class = "🟢 Low Risk"
            elif risk_score >= 4:
                risk_class = "🟡 Medium Risk"
            else:
                risk_class = "🔴 High Risk"

            # Display result
            st.success("✅ Prediction Complete!")
            st.markdown(f"**Risk Score:** `{risk_score:.2f}` *(0 = Very High Risk, 10 = Very Low Risk)*")
            st.markdown(f"**Risk Classification:** {risk_class}")
            st.markdown(f"**Probability of Default:** `{prob_default:.2%}`")

        except Exception as e:
            st.error(f"❌ Error: {e}")
