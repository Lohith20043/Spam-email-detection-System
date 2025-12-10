import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
)

# -------------------------------------------------------
# CUSTOM CSS
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
# SESSION HISTORY
# -------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------
# LOAD YOUR EXISTING MODELS
# -------------------------------------------------------
model_nb = joblib.load("naive_bayes_model.joblib")
model_lr = joblib.load("logistic_regression_model.joblib")
model_svm = joblib.load("svm.joblib")
vectorizer = joblib.load("vectorizer.joblib")


# -------------------------------------------------------
# PREDICTION FUNCTIONS
# -------------------------------------------------------
def preprocess(email):
    return vectorizer.transform([email])

def predict_nb(email):
    X = preprocess(email)
    pred = model_nb.predict(X)[0]
    conf = model_nb.predict_proba(X)[0][1]
    return pred, conf

def predict_lr(email):
    X = preprocess(email)
    conf = model_lr.predict_proba(X)[0][1]
    pred = 1 if conf >= 0.50 else 0
    return pred, conf

def predict_svm(email):
    X = preprocess(email)
    pred = model_svm.predict(X)[0]
    score = model_svm.decision_function(X)[0]
    conf = 1 / (1 + np.exp(-score))
    return pred, conf


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "History", "About"],
    index=0
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if menu == "Home":
    st.markdown("<h1 class='main-title'>📧 Spam Email Detection Portal</h1>", unsafe_allow_html=True)
    st.write("Enter the email content below to analyze spam using Naive Bayes, Logistic Regression, and SVM.")

    email_input = st.text_area("✍️ Enter Email Content")

    if st.button("🔍 Analyze"):
        if email_input.strip() == "":
            st.warning("Please enter email content.")
        else:
            # Predictions from 3 models
            nb_pred, nb_conf = predict_nb(email_input)
            lr_pred, lr_conf = predict_lr(email_input)
            svm_pred, svm_conf = predict_svm(email_input)

            spam_votes = nb_pred + lr_pred + svm_pred
            avg_conf_spam = (nb_conf + lr_conf + svm_conf) / 3
            avg_conf_not = (1 - nb_conf + 1 - lr_conf + 1 - svm_conf) / 3

            # Final decision by voting
            if spam_votes >= 2:
                final = "SPAM"
                conf = avg_conf_spam
                st.error(f"🚨 Final Decision: **SPAM** ({round(conf*100,2)}%)")
            else:
                final = "NOT SPAM"
                conf = avg_conf_not
                st.success(f"✅ Final Decision: **NOT SPAM** ({round(conf*100,2)}%)")

            st.markdown("---")
            st.subheader("📊 Model-wise Predictions")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="card">
                    <h4 class="sub-header">🧠 Naive Bayes</h4>
                    <b>{'Spam' if nb_pred else 'Not Spam'}</b><br>
                    Confidence: {round(nb_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card">
                    <h4 class="sub-header">📈 Logistic Regression</h4>
                    <b>{'Spam' if lr_pred else 'Not Spam'}</b><br>
                    Confidence: {round(lr_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card">
                    <h4 class="sub-header">📊 SVM</h4>
                    <b>{'Spam' if svm_pred else 'Not Spam'}</b><br>
                    Confidence: {round(svm_conf*100,2)} %
                </div>
                """, unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "email": email_input,
                "result": final,
                "confidence": round(conf * 100, 2),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })


# -------------------------------------------------------
# HISTORY PAGE
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
# ABOUT PAGE
# -------------------------------------------------------
else:
    st.header("ℹ️ About This Project")
    st.write("""
This system uses **three ML models** trained on your dataset:

- 🧠 Naive Bayes  
- 📈 Logistic Regression  
- 📊 Support Vector Machine (SVM)  

It performs:

✔ TF-IDF Vectorization  
✔ 3-Model Voting System  
✔ Confidence Score Calculation  
✔ Email History Tracking  
    """)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("<div class='footer'>👨‍💻 Developed by Team | Spam Detection ML Project</div>", unsafe_allow_html=True)
