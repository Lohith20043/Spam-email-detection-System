import streamlit as st
import joblib
from datetime import datetime

# -------------------------------------------------------
# 🔹 PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
)

# -------------------------------------------------------
# 🔹 CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 45px !important;
            color: #4A90E2;
            font-weight: bold;
        }
        .sub-header {
            color: #0a66c2;
        }
        .card {
            padding: 15px 20px;
            background-color: #000000;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        }
        .history-box {
            background: #000000;
            padding: 10px;
            border-radius: 12px;
            font-size: 15px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            padding-top: 20px;
            opacity: 0.7;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🔹 SESSION HISTORY
# -------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------
# 🔹 LOAD TRAINED MODEL
# -------------------------------------------------------
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def classify_email(email):
    x = vectorizer.transform([email])
    pred = model.predict(x)[0]     # 1 = spam, 0 = not spam
    conf = model.predict_proba(x)[0][1]  # spam probability
    return pred, conf


# -------------------------------------------------------
# 🔹 SIDEBAR NAVIGATION
# -------------------------------------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "History", "About"],
    index=0
)

# -------------------------------------------------------
# 🔹 HOME PAGE
# -------------------------------------------------------
if menu == "Home":
    st.markdown("<h1 class='main-title'>📧 Spam Email Detection Portal</h1>", unsafe_allow_html=True)
    st.write("Enter any email text below and get spam detection with confidence score.")

    email_input = st.text_area("✍️ Enter Email Content")

    if st.button("🔍 Analyze"):
        if email_input.strip() == "":
            st.warning("Please enter valid email content!")
        else:
            pred, conf = classify_email(email_input)

            if pred == 1:
                result = "SPAM"
                st.error(f"🚨 Final Decision: **SPAM** ({round(conf*100,2)}%)")
            else:
                result = "NOT SPAM"
                st.success(f"✅ Final Decision: **NOT SPAM** ({round((1-conf)*100,2)}%)")

            # Save history
            st.session_state.history.append({
                "email": email_input,
                "result": result,
                "confidence": round(conf*100, 2),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

# -------------------------------------------------------
# 🔹 HISTORY PAGE
# -------------------------------------------------------
elif menu == "History":
    st.header("📜 Analysis History")

    if len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"""
                <div class="history-box">
                <b>{i}. {item['result']} ({item['confidence']}%)</b><br>
                <i>{item['email'][:200]}...</i><br>
                <small>Time: {item['time']}</small>
                </div><br>
            """, unsafe_allow_html=True)

# -------------------------------------------------------
# 🔹 ABOUT PAGE
# -------------------------------------------------------
else:
    st.header("ℹ️ About the Spam Detector")
    st.write("""
This system uses Machine Learning (Naive Bayes + TF-IDF)  
to classify emails as **Spam** or **Not Spam**.

### ✔ Features
- Fast and accurate spam detection  
- Custom training using your own examples  
- Confidence score calculation  
- Clean UI with history tracking

### ✔ Technology Used
- Python  
- Scikit-learn  
- Streamlit  
- Naive Bayes Model  
- TF-IDF Vectorizer  
    """)

# -------------------------------------------------------
# 🔹 FOOTER
# -------------------------------------------------------
st.markdown("<div class='footer'>👨‍💻 Developed by Our Team | ML Spam Detection Tool</div>", unsafe_allow_html=True)
