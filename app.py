import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import os
from datetime import datetime

# ----------------------------
# Setup Groq Client
# ----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ----------------------------
# App Title
# ----------------------------
st.title("💸 AI Expense Analyzer")
st.write("Track expenses, view charts, and get smart AI-powered spending insights.")

# ----------------------------
# Session State Initialization
# ----------------------------
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Amount", "Category", "Description", "Date"])

# ----------------------------
# Expense Input Form
# ----------------------------
st.header("➕ Add Expense")

with st.form("expense_form"):
    amount = st.number_input("Expense Amount", min_value=1.0, format="%.2f")
    description = st.text_input("Description (e.g., Dinner at Italian restaurant)")
    date = st.date_input("Date", value=datetime.today())
    submit = st.form_submit_button("Add Expense")

# ----------------------------
# LLM Categorization
# ----------------------------
def categorize_expense(description):
    """Use Groq LLM to infer expense category."""
    prompt = f"""
    The user entered this expense description: "{description}"

    Classify the category. Choose one from:
    Food, Groceries, Travel, Shopping, Entertainment, Bills, Health, Education, Other.

    Return ONLY the category name.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    return response.choices[0].message.content.strip()

# ----------------------------
# Add Expense to Table
# ----------------------------
if submit:
    if description.strip() == "":
        st.error("Please enter a description.")
    else:
        category = categorize_expense(description)
        new_row = pd.DataFrame(
            [[amount, category, description, str(date)]],
            columns=["Amount", "Category", "Description", "Date"]
        )
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
        st.success(f"Expense added! Category detected: **{category}**")

# ----------------------------
# Expense Table
# ----------------------------
st.header("📊 All Expenses")
st.dataframe(st.session_state.expenses)

# ----------------------------
# Charts Section
# ----------------------------
if not st.session_state.expenses.empty:
    st.header("📈 Expense Charts")

    # Bar Chart by Category
    category_totals = st.session_state.expenses.groupby("Category")["Amount"].sum()

    st.subheader("Bar Chart — Category-wise Spending")
    fig1, ax1 = plt.subplots()
    ax1.bar(category_totals.index, category_totals.values)
    ax1.set_ylabel("Amount")
    ax1.set_xlabel("Category")
    ax1.set_title("Total Spending by Category")
    st.pyplot(fig1)

    # Pie Chart
    st.subheader("Pie Chart — Spending Distribution")
    fig2, ax2 = plt.subplots()
    ax2.pie(category_totals.values, labels=category_totals.index, autopct="%1.1f%%")
    ax2.set_title("Expense Distribution")
    st.pyplot(fig2)

# ----------------------------
# AI Spending Analysis
# ----------------------------
st.header("🤖 AI Expense Insights")

if st.button("Generate Insights"):
    if st.session_state.expenses.empty:
        st.warning("Add some expenses first!")
    else:
        df = st.session_state.expenses

        # Prepare summary for LLM
        expense_text = df.to_string(index=False)

        prompt = f"""
        Analyze the following expense data:

        {expense_text}

        Provide:
        1. Key spending patterns.
        2. Categories where the user overspends.
        3. Monthly or weekly spending trends.
        4. Personalized strategies to manage expenses.
        5. Any anomalies or sudden spikes.

        Keep the explanation clear, structured, and actionable.
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )

        insights = response.choices[0].message.content
        st.write(insights)
