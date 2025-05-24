import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

def generate_pdf_report(user_responses, result_dict, ai_report_text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf_path = tmp_file.name

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("🧠 Social Media Impact Assessment", styles["Title"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("📋 User Responses:", styles["Heading2"]))
    for question, answer in user_responses.items():
        elements.append(Paragraph(f"<b>{question}</b>: {answer}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🎯 AI-Predicted Scores:", styles["Heading2"]))
    for key, value in result_dict.items():
        elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("🧠 Gemini AI Psychological Report:", styles["Heading2"]))
    elements.append(Paragraph(ai_report_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    doc.build(elements)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    os.remove(pdf_path)

    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="mental_health_report.pdf">📄 Download PDF Report</a>'
    return href



# Page config
st.set_page_config(page_title="Social Media Impact Analysis Chatbot", layout="wide")

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("socialMediaImpact.csv")

    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].astype('int64') // 10**9
        except:
            pass

    imputer = SimpleImputer(strategy='most_frequent')
    df.iloc[:, :] = imputer.fit_transform(df)
    return df, imputer

df, imputer = load_data()

# Target columns
target_columns = [
    "18. How often do you feel depressed or down?",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?"
]

# Question columns
question_columns = [col for col in df.columns if col not in target_columns]


# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
X = df.drop(columns=target_columns)
y = df[target_columns]
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar - Choose model
st.sidebar.title("🔧 Select ML Algorithm")
model_choice = st.sidebar.selectbox("Choose a Machine Learning Model", ["Logistic Regression", "Random Forest", "SVM", "XGBoost"])

# Train model
if model_choice == "Logistic Regression":
    base_model = LogisticRegression(class_weight="balanced")
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test_eval = y_test

elif model_choice == "Random Forest":
    base_model = RandomForestClassifier(class_weight="balanced")
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test_eval = y_test

elif model_choice == "SVM":
    base_model = SVC()
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test_eval = y_test

elif model_choice == "XGBoost":
    from xgboost import XGBClassifier
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model = MultiOutputClassifier(base_model)
    y_train_adj = y_train - 1
    y_test_adj = y_test - 1
    model.fit(X_train, y_train_adj)
    predictions_adj = model.predict(X_test)
    predictions = predictions_adj + 1
    y_test_eval = y_test_adj + 1

# Display metrics
st.sidebar.subheader("📊 Model Evaluation")
for i, col in enumerate(target_columns):
    acc = accuracy_score(y_test_eval.iloc[:, i], predictions[:, i])
    prec = precision_score(y_test_eval.iloc[:, i], predictions[:, i], average='weighted', zero_division=0)
    rec = recall_score(y_test_eval.iloc[:, i], predictions[:, i], average='weighted', zero_division=0)
    st.sidebar.markdown(f"**{col}**")
    st.sidebar.markdown(f"Accuracy: {acc:.2f}")
    st.sidebar.markdown(f"Precision: {prec:.2f}")
    st.sidebar.markdown(f"Recall: {rec:.2f}")
    # Consider using F1-score or MAE for better insight on Likert scale

# Streamlit UI styling
st.markdown("""
<style>
    body {background-color: #8B3A62; color: white; font-family: Arial;}
    .stButton > button {background-color: transparent; color: white; border-radius: 10px; border-color:white;padding: 10px;}
    .stChatMessage {padding: 10px; border-radius: 10px; max-width: 70%; display: block; word-wrap: break-word; background-color: gold;}
    .stChatMessage.system {background-color: white; color: black; float: left; clear: both;}
    .stChatMessage.user {background-color: navy; color: white; float: right; clear: both;}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Social Media Impact Analysis Chatbot 🌸")
st.write("Check Your mental health condition in accordance with impact of Social Media in your daily life")

# Chatbot state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
user_responses = {}

# Chatbot logic
def chatbot_response(user_input):
    current_q = question_columns[st.session_state.current_question]
    user_responses[current_q] = user_input
    st.session_state.current_question += 1

    if st.session_state.current_question < len(question_columns):
        next_q = question_columns[st.session_state.current_question]
        options = label_encoders[next_q].classes_.tolist() if next_q in label_encoders else []
        return next_q, options
    else:
        return "All questions answered! Click 'Submit' for assessment.", []

user_input = st.chat_input("Type your response here...")
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    bot_reply, options = chatbot_response(user_input)
    st.session_state.chat_history.append(("Bot", bot_reply))

    if options:
        st.write("### Suggested Responses:")
        st.write(options)

for sender, message in st.session_state.chat_history:
    align_class = "user" if sender == "You" else "system"
    st.markdown(f'<div class="stChatMessage {align_class}">{message}</div>', unsafe_allow_html=True)

# Submit button
if st.session_state.current_question >= len(question_columns) and st.button("✨ Submit and Get Assessment ✨"):
    user_input_dict = {q: user_responses.get(q, np.nan) for q in question_columns}
    for q in target_columns:
        user_input_dict[q] = 3

    user_input_df = pd.DataFrame([user_input_dict])
    for col in df.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0

    user_input_df = user_input_df[df.columns]
    for q in label_encoders:
        if q in user_input_df and user_input_df[q].notna().all():
            le = label_encoders[q]
            user_input_df[q] = user_input_df[q].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    user_input = user_input_df.to_numpy()
    user_input_imputed = imputer.transform(user_input)
    user_input_imputed_scaled = scaler.transform(user_input_imputed[:, :len(question_columns)])

    if model_choice == "XGBoost":
        prediction = model.predict(user_input_imputed_scaled) + 1
    else:
        prediction = model.predict(user_input_imputed_scaled)

    result = dict(zip(target_columns, prediction[0]))
    st.success("Assessment complete! 🎉")
    st.json(result)

    st.subheader("📊 Visualizations Based on Your Assessment")

    # Interactive bar chart
    bar_fig = px.bar(
    x=list(result.keys()),
    y=list(result.values()),
    labels={'x': 'Mental Health Metric', 'y': 'Predicted Score'},
    title="Your Mental Health Assessment Scores",
    color=list(result.values()),
    color_continuous_scale='RdPu',
    height=400
    )
    bar_fig.update_traces(marker_line_color='black', marker_line_width=1.5)
    st.plotly_chart(bar_fig, use_container_width=True)

    # Animated bubble chart to show score severity
    bubble_fig = px.scatter(
        x=list(result.keys()),
        y=list(result.values()),
        size=[val**2 for val in result.values()],
        color=list(result.values()),
        labels={'x': 'Assessment Category', 'y': 'Score'},
        title="Severity Representation via Bubble Size",
        size_max=60,
        color_continuous_scale='Plasma',
    )
    bubble_fig.update_layout(
        transition_duration=500,
        yaxis=dict(range=[0, 6]),
    )
    st.plotly_chart(bubble_fig, use_container_width=True)



    # Correlation heatmap of predictions vs actuals
    fig2, ax2 = plt.subplots()
    df_eval = pd.DataFrame(predictions, columns=target_columns)
    sns.heatmap(df_eval.corr(), annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Between Predicted Outcomes")
    st.pyplot(fig2)

    # Distribution plots of each prediction column
    for i, col in enumerate(target_columns):
        fig, ax = plt.subplots()
        sns.countplot(x=predictions[:, i], palette="viridis", ax=ax)
        ax.set_title(f"Distribution of Predictions for: {col}")
        ax.set_xlabel("Predicted Level")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Gemini AI Analysis
    st.subheader("📜 Detailed Mental Report")
    genai.configure(api_key="AIzaSyBva2qqvNun5SYuXAqI-pmGmot_n5U4MH0")
    model_gemini = genai.GenerativeModel("gemini-1.5-pro-latest")
    user_responses_text = "\n".join([f"{q}: {user_responses[q]}" for q in question_columns if q in user_responses])
    prompt = f"Analyze the following user responses and generate a psychological assessment:\n{user_responses_text}\nPredictions: {result}"
    response = model_gemini.generate_content(prompt)
    st.write(response.text)

    # PDF Download Button
    pdf_link = generate_pdf_report(user_responses, result, response.text)
    st.markdown("### 📥 Download Your Full AI Report")
    st.markdown(pdf_link, unsafe_allow_html=True)

