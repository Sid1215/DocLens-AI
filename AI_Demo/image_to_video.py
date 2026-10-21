import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN was not found in your .env file.")
    st.stop()


# ==========================================
# STREAMLIT
# ==========================================

st.set_page_config(
    page_title="AI Superhero Video Generator",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Superhero Video Generator")

st.write(
    "Create cinematic superhero videos using "
    "LangChain + Hugging Face."
)


# ==========================================
# INPUTS
# ==========================================

scene = st.text_area(
    "🎭 Describe your scene",
    value="""
A powerful futuristic superhero standing on top
of a skyscraper during a stormy night in a
futuristic city, wearing an advanced armored
superhero suit.
""",
    height=130
)


effects = st.text_area(
    "⚡ Describe the VFX",
    value="""
Glowing energy around both hands,
blue and red energy trails,
electric lightning,
floating sparks,
smoke and debris,
glowing eyes,
dramatic clouds,
energy explosion behind the superhero.
""",
    height=130
)


# ==========================================
# LANGCHAIN PROMPT
# ==========================================

prompt_template = PromptTemplate.from_template(
    """
Create a cinematic superhero video.

Scene:
{scene}

VFX:
{effects}

Visual style:
cinematic superhero blockbuster,
photorealistic,
realistic CGI,
dramatic lighting,
volumetric smoke,
glowing particles,
detailed futuristic city,
high contrast,
premium cinematic visual effects.

Motion:
The superhero moves naturally.
Energy flows around the hands.
Lightning flashes in the sky.
Smoke moves through the environment.
Sparks fly through the air.
The camera slowly moves toward the superhero.

Make the scene dynamic and cinematic.
"""
)


# ==========================================
# GENERATE VIDEO
# ==========================================

if st.button("🎬 Generate Video", type="primary"):

    if not scene.strip():
        st.warning("Please describe your scene.")
        st.stop()

    if not effects.strip():
        st.warning("Please describe the VFX.")
        st.stop()


    # --------------------------------------
    # Create prompt
    # --------------------------------------

    prompt = prompt_template.format(
        scene=scene,
        effects=effects
    )


    # --------------------------------------
    # Show prompt
    # --------------------------------------

    with st.expander("🔎 View Generated Prompt"):
        st.write(prompt)


    # --------------------------------------
    # Hugging Face client
    # --------------------------------------

    client = InferenceClient(
        provider="fal-ai",
        api_key=HF_TOKEN
    )


    # --------------------------------------
    # Generate video
    # --------------------------------------

    with st.spinner(
        "🎬 Generating video... Please wait."
    ):

        try:

            video_bytes = client.text_to_video(
                prompt,
                model="Wan-AI/Wan2.2-TI2V-5B"
            )

        except Exception as e:

            st.error("❌ Video generation failed.")

            st.exception(e)

            st.stop()


    # --------------------------------------
    # Check result
    # --------------------------------------

    if not video_bytes:

        st.error(
            "❌ Hugging Face returned an empty video."
        )

        st.stop()


    # --------------------------------------
    # Display
    # --------------------------------------

    st.success("🎉 Video generated successfully!")

    st.video(video_bytes)


    # --------------------------------------
    # Download
    # --------------------------------------

    st.download_button(
        label="⬇️ Download Video",
        data=video_bytes,
        file_name="superhero_vfx.mp4",
        mime="video/mp4"
    )