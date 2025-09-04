import os
import base64
import streamlit as st
from dotenv import load_dotenv
import backend_langchain as backend

load_dotenv()

st.set_page_config(
    page_title="Restaurant Details Generator",
    page_icon="logo.png",
    layout="centered"
)

def set_background_image(png_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{png_file}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

set_background_image("https://raw.githubusercontent.com/Kratugautam99/Agentic-AI-Learning/refs/heads/master/LangChain_and_LangGraph/RestaurantDetailsGenerator/restaurant.png")
local_css("style.css")

if "result" not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown('<div class="bordered-img"><img src="https://raw.githubusercontent.com/Kratugautam99/Agentic-AI-Learning/refs/heads/master/LangChain_and_LangGraph/RestaurantDetailsGenerator/logo.png" width="120"></div>',unsafe_allow_html=True)
with col2:
    st.markdown(
        """
        <div class="overlay-container-1">
            <h2>Restaurant Name & Menu Generator</h2>
            <p>Powered by LangChain + Gemini</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.header("Settings")
    cuisine = st.selectbox(
        "Pick a cuisine",
        (   "African",
            "Algerian",
            "American",
            "Argentinian",
            "Armenian",
            "Asian Fusion",
            "Australian",
            "Austrian",
            "Bangladeshi",
            "Belgian",
            "Brazilian",
            "British",
            "Bulgarian",
            "Burmese",
            "Cambodian",
            "Canadian",
            "Caribbean",
            "Chilean",
            "Chinese - Cantonese",
            "Chinese - Sichuan",
            "Chinese - Hunan",
            "Chinese - Northern",
            "Chinese - Shanghai",
            "Chinese - Dim Sum",
            "Colombian",
            "Cuban",
            "Czech",
            "Danish",
            "Dominican",
            "Ecuadorean",
            "Egyptian",
            "Ethiopian",
            "Filipino",
            "Finnish",
            "French",
            "Georgian",
            "German",
            "Greek",
            "Hawaiian",
            "Hungarian",
            "Iberian",
            "Indian - North Indian",
            "Indian - South Indian",
            "Indian - Punjabi",
            "Indian - Gujarati",
            "Indian - Bengali",
            "Indian - Konkani",
            "Indonesian",
            "Iranian (Persian)",
            "Iraqi",
            "Irish",
            "Israeli",
            "Italian",
            "Japanese",
            "Jordanian",
            "Korean",
            "Kurdish",
            "Laotian",
            "Lebanese",
            "Libyan",
            "Lithuanian",
            "Malaysian",
            "Maltese",
            "Mexican",
            "Moroccan",
            "Nepalese",
            "New Zealand",
            "Nigerian",
            "Norwegian",
            "Omani",
            "Pakistani",
            "Palestinian",
            "Peruvian",
            "Polish",
            "Portuguese",
            "Puerto Rican",
            "Russian",
            "Salvadoran",
            "Saudi",
            "Scandinavian",
            "Scottish",
            "Senegalese",
            "Serbian",
            "Singaporean",
            "Slovak",
            "Slovenian",
            "Somali",
            "South African",
            "South American",
            "Spanish",
            "Sri Lankan",
            "Sudanese",
            "Swedish",
            "Swiss",
            "Syrian",
            "Taiwanese",
            "Tanzanian",
            "Thai",
            "Turkish",
            "Tunisian",
            "Ugandan",
            "Ukrainian",
            "Uruguayan",
            "Uzbek",
            "Venezuelan",
            "Vietnamese",
            "Yemeni",
            "Zambian",
            "Zimbabwean",
            "Fusion",
            "Modern European",
            "Mediterranean",
            "Vegan",
            "Vegetarian",
            "Gluten-Free",
            "Healthy",
            "Street Food",
            "Brunch",
            "Dessert"
                            ),
                                index=0,
                                        )
    custom_cuisine = st.text_input("Or type a custom cuisine", placeholder="e.g. Korean BBQ, Indo-Sino Food")
    model_name = st.selectbox(
        "Model",
        ("Gemini-2.5-Flash", "Gemini-1.5-Flash", "Gemini-1.5-Pro"),
        index=0
    )
    temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
    st.markdown("---")
    st.write("API key status:")
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if has_key:
        st.success("Google API key found")
    else:
        st.error("No Google API key")

col1, col2 = st.columns([1, 1])
with col1:
    generate = st.button("Generate Restaurant", type="primary", width="stretch")
with col2:
    clear = st.button("Clear Results", width="stretch")

selected_cuisine = custom_cuisine.strip() if custom_cuisine.strip() else cuisine

if clear:
    st.session_state.result = None
    st.rerun()

if generate:
    if not has_key:
        st.error("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")
    else:
        with st.spinner("Cooking up a brand identity..."):
            try:
                st.session_state.result = backend.generate_restaurant_name_and_items(
                    cuisine=selected_cuisine,
                    model_name=model_name.lower(),
                    temperature=temperature
                )
            except Exception as e:
                st.session_state.result = None
                st.error(f"Generation failed: {e}")

if st.session_state.result:
    name = st.session_state.result.get("restaurant_name")
    items = st.session_state.result.get("menu_items")
    
    if name:
        st.markdown(f'<div class="overlay-container-2"><h3>{name}</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="overlay-container-2"><h3>Menu Items</h3></div>', unsafe_allow_html=True)
        if items:
            for item in items:
                st.markdown(f'<div class="overlay-container-2"><p>- {item}</p></div>', unsafe_allow_html=True)
        else:
            raw = st.session_state.result.get("menu_items_raw", "")
            if raw:
                st.code(raw, language="markdown")
            else:
                st.info("No menu items returned. Try increasing creativity or changing the cuisine.")
    else:
        st.error("No restaurant name generated. Please try again.")