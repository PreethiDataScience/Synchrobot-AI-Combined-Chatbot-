import streamlit as st
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------
# Load OpenRouter API key
# -----------------------
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
st.sidebar.write("✅ OpenRouter key loaded:", bool(OPENROUTER_KEY))

# -----------------------
# Streamlit UI
# -----------------------
st.title("OpenRouter Models")
st.write("Enter your question below and press Enter:")

user_input = st.text_input("Ask a question:")

# -----------------------
# Initialize OpenRouter client
# -----------------------
if OPENROUTER_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )

# -----------------------
# Function to call model safely (no visible error)
# -----------------------
def safe_call_model(model_name, user_input, retries=3, delay=3):
    """Call a model safely, retrying on rate limits silently."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_input}],
                extra_headers={
                    "HTTP-Referer": "https://your-site.com",
                    "X-Title": "SynchroBot AI",
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):  # Rate limit
                time.sleep(delay)
            else:
                # Don’t show red box, just log quietly
                print(f"⚠️ {model_name} failed: {e}")
                return ""
    return ""

# -----------------------
# Main logic
# -----------------------
if user_input and OPENROUTER_KEY:
    with st.spinner("Generating answer from both models..."):
        # Model 1 (DeepSeek r1-0528)
        model1_answer = safe_call_model("deepseek/deepseek-r1-0528:free", user_input)

        # Model 2 (DeepSeek r1-0528-qwen3-8b)
        model2_answer = safe_call_model("deepseek/deepseek-r1-0528-qwen3-8b:free", user_input)

    # Combine the results
    if model1_answer or model2_answer:
        st.write("### Combined Answer")
        if model1_answer:
            st.markdown(f" {model1_answer}")
        if model2_answer:
            st.markdown(f"{model2_answer}")
    else:
        st.warning("No responses received. Please try again later.")
