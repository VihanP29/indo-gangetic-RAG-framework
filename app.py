import streamlit as st
from rag_backend import ask_igp

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="IGP.ai",
    page_icon="🌾",
    layout="centered"
)

# ----------------------------
# CUSTOM CSS (INDO-GANGETIC THEME)
# ----------------------------
st.markdown("""
<style>

/* === APP BACKGROUND GRADIENT === */
.stApp {
    background: linear-gradient(
        135deg,
        #F5F9F7 0%,
        #E3F2EC 25%,
        #D6EAF8 50%,
        #FDF6EC 75%,
        #F5F9F7 100%
    );
    font-family: 'Segoe UI', sans-serif;
}

/* === MAIN HEADER === */
h1 {
    color: #1F6AE1;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* === SUBTITLE === */
.subtitle {
    color: #2FA36B;
    font-size: 18px;
    margin-bottom: 10px;
    font-weight: 500;
}

/* === REGION HINT === */
.smallhint {
    color: #4A6F73;
    font-size: 14px;
    margin-bottom: 20px;
}

/* === INPUT BOX === */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #CFAE70 !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* === CHAT RESPONSE CARD === */
.chatbox {
    background: linear-gradient(145deg, #ffffff, #F2FBF7);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    border-left: 6px solid #2FA36B;
    margin-top: 12px;
    font-size: 16px;
    line-height: 1.6;
}

/* === BUTTON STYLE === */
button {
    background: linear-gradient(135deg, #1F6AE1, #2FA36B) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 10px 18px !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1>🌾 IGP.ai</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Indo-Gangetic Agricultural Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='smallhint'>India • Pakistan • Nepal • Bangladesh</div>", unsafe_allow_html=True)

# ----------------------------
# USER INPUT
# ----------------------------
user_query = st.text_input("Ask a question about crops, soil, or climate:")

# ----------------------------
# RESPONSE
# ----------------------------
if user_query:
    with st.spinner("Consulting Indo-Gangetic knowledge base..."):
        answer = ask_igp(user_query)

    st.markdown("### 🤖 Response")
    st.markdown(f"<div class='chatbox'>{answer}</div>", unsafe_allow_html=True)
