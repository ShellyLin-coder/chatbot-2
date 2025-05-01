import streamlit as st
import google.generativeai as genai
from datetime import datetime
import csv
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
import numpy as np
import re
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import uuid
import joblib  # or use pickle
from sklearn.feature_extraction.text import TfidfVectorizer



# ---- Load Logistic Regression Model ----
pipeline = joblib.load("logreg_model.pkl")  # ✅ this pipeline includes vectorizer + model


# ---- Load ID to Label Mapping ----
id2label = joblib.load("id2label.pkl")  # Assuming this is a dict like {0: 'SuicideWatch', 1: 'depression', 2: 'teenagers'}




# ---- Label Maps ----
label_map = {0: "SuicideWatch", 1: "depression", 2: "teenagers"}
raw_to_display = {"SuicideWatch": "Suicide", "depression": "Depression", "teenagers": "Teenager"}

# ---- Preprocessing ----
def clean_input(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ---- Classify Text ----
@st.cache_data(show_spinner=False)
def classify_all_prompts(df):
    def classify_text(text):
        text = clean_input(text)
        prediction = pipeline.predict([text])[0]
        raw_label = id2label.get(prediction, str(prediction))
        return raw_to_display.get(raw_label, raw_label)
    
    df["prediction"] = df["prompt"].astype(str).apply(classify_text)
    return df




# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("UIC_BUSINESS_LOGO.PNG", width=180)
    st.markdown("""<h4 style='text-align: center;'>Group 7: Drashti Bhinde, Chia-Hsuan Lin, Bo Pang, Tang-Hua Chen, Yi-Hsuan Tseng</h4><hr style='border:1px solid #f4f4f4;'>""", unsafe_allow_html=True)
    st.markdown("## 📚 Navigation")
    MENU_OPTIONS = {"Chatbot": "💬 Chatbot", "Dashboard": "📊 Dashboard"}
    for page_id, page_label in MENU_OPTIONS.items():
        if st.button(page_label):
            st.session_state.page = page_id
            if page_id != "Dashboard":
                st.session_state.authenticated = False
    st.markdown("---")
    
    gemini_api_key = st.secrets["api_keys"]["gemini"]
    st.markdown("[Get Gemini API Key](https://makersuite.google.com/app/apikey)")

if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

selected_page = st.session_state.page

# ---------------- Chatbot Page ----------------
if selected_page == "Chatbot":
    col1, col2 = st.columns([9, 1])
    with col2:
        st.image("Soultalk logo.png", width=130)
    st.title("🌸 Welcome to SoulTalk")
    st.caption('Your Mental Wellness Buddy - Powered by Google Gemini and DistilBERT')

    if "seen_disclaimer" not in st.session_state:
        st.session_state.seen_disclaimer = False

    if not st.session_state.seen_disclaimer:
        st.warning("⚠️ **Disclaimer**: Your input may be stored and analyzed for research or improvement purposes.")
        if st.button("✅ I Understand"):
            st.session_state.seen_disclaimer = True

    if st.session_state.seen_disclaimer:
        if "user_id" not in st.session_state or not st.session_state.user_id:
            st.session_state.user_id = st.text_input("Enter your User ID to start:", key="user_id_input")
        if not st.session_state.user_id:
            st.stop()
    else:
        st.stop()

    if st.sidebar.button("🔄 Clear Chat"):
        st.session_state.chat_history = [("assistant", "Hello, I'm here for you. How are you feeling today?")]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [("assistant", "Hello, I'm here for you. How are you feeling today?")]

    for role, content in st.session_state.chat_history:
        st.chat_message(role).write(content)

    if prompt := st.chat_input("Please enter your thoughts or feelings..."):
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key.")
            st.stop()

        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").write(prompt)

        cleaned_prompt = clean_input(prompt)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("user_input_log.csv", mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([st.session_state.user_id, timestamp, cleaned_prompt])

        try:
            genai.configure(api_key=gemini_api_key)
            model_name = "gemini-2.0-flash"
            gemini_model = genai.GenerativeModel(model_name)
            chat = gemini_model.start_chat(history=[])
            system_prompt = ("You are a kind and empathetic mental health support assistant. "
                             "Respond gently, and give short encouraging advice when needed.")

            if "pro" in model_name:
                chat = gemini_model.start_chat(history=[], system_instruction=system_prompt)
            else:
                chat.send_message(system_prompt)

            response = chat.send_message(prompt)
            reply = response.text

            st.session_state.chat_history.append(("assistant", reply))
            st.chat_message("assistant").write(reply)

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# ---------------- Dashboard Page ----------------
elif selected_page == "Dashboard":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username == "localhost" and password == "Demo1234":
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                else:
                    st.error("❌ Invalid credentials")
    else:
        col1, col2 = st.columns([9, 1])
        with col2:
            st.image("Soultalk logo.png", width=120)
        st.title("📊 SoulTalk Dashboard")
        st.success("🔓 Logged in")

        if st.button("🚪 Log out"):
            st.session_state.authenticated = False
            st.rerun()

        try:
            df = pd.read_csv("user_input_log.csv", names=["user_id", "timestamp", "prompt"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = classify_all_prompts(df)

            tab1, tab2, tab3 = st.tabs(["Table 📋", "Stats 📈", "Word Cloud ☁️"])

            with tab1:
                st.subheader("📋 User Responses")
                st.dataframe(df[["user_id", "timestamp", "prompt", "prediction"]])

                st.subheader("📈 Prediction Class Distribution")
                class_dist = df["prediction"].value_counts().reset_index()
                class_dist.columns = ["Class", "Count"]
                bar = alt.Chart(class_dist).mark_bar().encode(
                    x=alt.X("Class", sort=None),
                    y="Count"
                )
                st.altair_chart(bar, use_container_width=True)

            with tab2:
                view_option = st.radio("Select Time Granularity:", ["Minute", "Hour", "Date"], horizontal=True)
                if view_option == "Minute":
                    df_grouped = df.groupby(df["timestamp"].dt.floor("min"))["prompt"].count().reset_index()
                elif view_option == "Hour":
                    df_grouped = df.groupby(df["timestamp"].dt.floor("h"))["prompt"].count().reset_index()
                else:
                    df_grouped = df.groupby(df["timestamp"].dt.date)["prompt"].count().reset_index()

                df_grouped.rename(columns={"prompt": "Messages", df_grouped.columns[0]: "time"}, inplace=True)

                st.subheader(f"📈 Message Count Over Time (by {view_option})")
                line = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X("time:T", title=view_option),
                    y=alt.Y("Messages:Q", title="Message Count", axis=alt.Axis(tickMinStep=1))
                ).properties(height=300)
                st.altair_chart(line, use_container_width=True)

                st.subheader("📏 Prompt Length Distribution")
                df["length"] = df["prompt"].astype(str).apply(len)
                df["length_bin"] = pd.cut(df["length"], bins=np.arange(0, df["length"].max() + 5, 5), right=False)
                df["length_bin_label"] = df["length_bin"].apply(lambda x: f"{int(x.left)+1}–{int(x.right)}")
                hist_data = df["length_bin_label"].value_counts().sort_index().reset_index()
                hist_data.columns = ["Prompt Length", "Count"]

                chart = alt.Chart(hist_data).mark_bar().encode(
                    x=alt.X("Prompt Length:N", title="Prompt Length", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1))
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

                st.subheader("🔤 Most Common Words")
                words = " ".join(df["prompt"].astype(str).tolist()).split()
                word_freq = Counter(words)
                common_words = pd.DataFrame(word_freq.most_common(10), columns=["word", "count"])
                bar = alt.Chart(common_words).mark_bar().encode(
                    y=alt.Y("word", sort="-x"),
                    x="count"
                )
                st.altair_chart(bar, use_container_width=True)

            with tab3:
                if not df.empty:
                    text = " ".join(df["prompt"].astype(str))
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("No text data available for Word Cloud.")

        except FileNotFoundError:
            st.warning("⚠️ No input log found yet.")