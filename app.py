import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from groq import Groq
from datetime import datetime

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AI Expense Analyzer", layout="wide")

# ----------------- INITIALIZATION -----------------
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Amount", "Category", "Description"])

# ----------------- GROQ SETUP -----------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("⚠️ Please set your GROQ_API_KEY environment variable in Colab.")
client = Groq(api_key=api_key)

# ----------------- UI HEADER -----------------
st.title("💸 AI Expense Analyzer")
st.markdown("Track your spending, visualize patterns, and get AI insights powered by **Groq** and **Streamlit**.")

# ----------------- ADD EXPENSE SECTION -----------------
with st.expander("➕ Add Expense", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        date = st.date_input("Date", datetime.now())
    with c2:
        amount = st.number_input("Amount (USD)", min_value=0.0, step=1.0)
    with c3:
        category = st.text_input("Category (e.g., Food, Transport, Rent)")
    desc = st.text_area("Description")

    if st.button("Add Expense"):
        new_row = pd.DataFrame([[date, amount, category, desc]], columns=["Date", "Amount", "Category", "Description"])
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
        st.success("✅ Expense added!")

# ----------------- SHOW EXPENSES -----------------
if not st.session_state.expenses.empty:
    df = st.session_state.expenses
    st.subheader("📊 Expense Table")
    st.dataframe(df, use_container_width=True)

    # Charts
    st.subheader("📈 Visualizations")
    t1, t2 = st.tabs(["Bar Chart", "Pie Chart"])

    with t1:
        fig, ax = plt.subplots()
        df.groupby("Category")["Amount"].sum().plot(kind="bar", ax=ax)
        ax.set_title("Total per Category")
        ax.set_ylabel("USD")
        st.pyplot(fig)

    with t2:
        fig, ax = plt.subplots()
        df.groupby("Category")["Amount"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Expense Distribution")
        st.pyplot(fig)

    # Trend
    st.subheader("📅 Daily Spending Trend")
    trend = df.groupby("Date")["Amount"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(trend["Date"], trend["Amount"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.set_title("Spending Over Time")
    st.pyplot(fig)

    # AI Analysis
    st.subheader("🤖 AI Insights via Groq")
    if st.button("Analyze My Spending"):
        with st.spinner("Groq analyzing your expenses..."):
            text_data = df.to_string(index=False)
            prompt = f"""
            Analyze these expenses. Identify spending patterns, overspending areas, and give 
            practical strategies to save money and balance the budget.

            Expenses:
            {text_data}
            """
            try:
                result = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                st.markdown("### 💡 AI Recommendations:")
                st.write(result.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq request failed: {e}")
else:
    st.info("No expenses yet. Add your first one above!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Groq")
