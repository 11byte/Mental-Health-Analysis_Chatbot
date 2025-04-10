import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import google.generativeai as genai

st.set_page_config(page_title="Social Media Impact Analysis Chatbot", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("socialMediaImpact.csv")

    # Convert any datetime strings
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].astype('int64') // 10**9
        except:
            continue

    # Fill missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df, imputer

df, imputer = load_data()

# Define targets
target_columns = [
    "18. How often do you feel depressed or down?",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?"
]
question_columns = [col for col in df.columns if col not in target_columns]

# Label encode categorical values
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df[question_columns]
y = df[target_columns]

# Ensure all values non-negative for MultinomialNB
X = np.where(X < 0, 1, X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
st.sidebar.title("🔧 Model: Multinomial Naive Bayes (as in working notebook)")
nb_model = MultiOutputClassifier(MultinomialNB(alpha=1.0))
nb_model.fit(X_train, y_train)
predictions = nb_model.predict(X_test)

# Evaluate
st.sidebar.subheader("📊 Model Evaluation")
for i, col in enumerate(target_columns):
    acc = accuracy_score(y_test.iloc[:, i], predictions[:, i])
    prec = precision_score(y_test.iloc[:, i], predictions[:, i], average='weighted', zero_division=0)
    rec = recall_score(y_test.iloc[:, i], predictions[:, i], average='weighted', zero_division=0)

    st.sidebar.markdown(f"**{col}**")
    st.sidebar.markdown(f"Accuracy: {acc:.2f}")
    st.sidebar.markdown(f"Precision: {prec:.2f}")
    st.sidebar.markdown(f"Recall: {rec:.2f}")

# UI Setup
st.title("🌸 Social Media Impact Analysis Chatbot 🌸")
st.write("Check your mental health condition in accordance with the impact of social media in your daily life.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
user_responses = {}

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

if st.session_state.current_question >= len(question_columns) and st.button("✨ Submit and Get Assessment ✨"):
    input_dict = {q: user_responses.get(q, np.nan) for q in question_columns}
    for q in target_columns:
        input_dict[q] = 3

    input_df = pd.DataFrame([input_dict])
    for col in df.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[df.columns]

    # Encode
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    user_input_np = input_df[question_columns].to_numpy()
    user_input_np = np.where(user_input_np < 0, 1, user_input_np)  # Clip for MultinomialNB

    prediction = nb_model.predict(user_input_np)
    result = dict(zip(target_columns, prediction[0]))

    st.success("Assessment complete! 🎉")
    st.json(result)

    # Visualizations
    st.subheader("📊 Your Mental Health Predictions")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=list(result.keys()), y=list(result.values()), palette="pastel", ax=ax1)
    ax1.set_ylim(1, 5)
    ax1.set_title("Predicted Mental Health Scores")
    st.pyplot(fig1)

    # Gemini AI Report
    st.subheader("🧠 Gemini-Based Psychological Analysis")
    genai.configure(api_key="AIzaSyBva2qqvNun5SYuXAqI-pmGmot_n5U4MH0")
    model_gemini = genai.GenerativeModel("gemini-1.5-pro-latest")
    user_text = "\n".join([f"{k}: {v}" for k, v in user_responses.items()])
    prompt = f"Analyze the following user responses and provide psychological insight:\n{user_text}\nPredicted outcomes: {result}"
    response = model_gemini.generate_content(prompt)
    st.write(response.text)

# Style
st.markdown("""
<style>
    .stChatMessage {padding: 10px; border-radius: 10px; max-width: 70%; word-wrap: break-word;}
    .stChatMessage.system {background-color: #e0e0e0; color: black; float: left; clear: both;}
    .stChatMessage.user {background-color: #007BFF; color: white; float: right; clear: both;}
</style>
""", unsafe_allow_html=True)
