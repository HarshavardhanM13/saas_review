import streamlit as st
import requests
import base64
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ReviewMagic AI",
    page_icon="🎨",
    layout="centered"
)

# --- BACKEND URL ---
# When you deploy, this will change to your live API URL
BACKEND_URL = "https://saas-review.onrender.com"

# --- UI DESIGN ---
st.title("🌟 AI Social Review Designer")
st.markdown("""
Convert your text reviews into stunning, high-quality Instagram posts using 
**Cloudflare Flux AI** via our secure FastAPI backend.
""")

st.divider()

# --- USER INPUT SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    review = st.text_area(
        "Customer Review",
        placeholder="Enter the review text here...",
        height=150
    )

with col2:
    author = st.text_input("Customer Name", placeholder="e.g. Harsh S.")
    style = st.selectbox(
        "Visual Style",
        ["Modern Minimalist", "Cyberpunk Night", "Luxury Gold", "Nature Zen", "Vintage Film"]
    )

# --- GENERATION LOGIC ---
if st.button("🚀 Generate AI Post", use_container_width=True):
    if not review or not author:
        st.warning("⚠️ Please provide both a review and an author name.")
    else:
        # Prepare the data for the FastAPI backend
        payload = {
            "review": review,
            "author": author,
            "style": style
        }
        
        with st.spinner("🧠 Backend is communicating with Cloudflare AI..."):
            try:
                # Send the POST request to FastAPI
                response = requests.post(BACKEND_URL, json=payload, timeout=70)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success"):
                        # Decode the Base64 image received from the backend
                        image_bytes = base64.b64decode(data["image_b64"])
                        
                        st.success("✅ Post generated successfully!")
                        
                        # Display the image
                        st.image(image_bytes, use_container_width=True, caption=f"Style: {style}")
                        
                        # Download Button
                        st.download_button(
                            label="📥 Download High-Res Post",
                            data=image_bytes,
                            file_name=f"ai_review_{author.replace(' ', '_')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error("Backend failed to return image data.")
                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to the Backend. Is `backend.py` running on port 8000?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- FOOTER ---
st.divider()
st.caption("Powered by FastAPI + Streamlit + Cloudflare Workers AI")