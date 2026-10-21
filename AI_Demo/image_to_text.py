import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI VFX Generator",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎬 AI Superhero image Generator")

st.write(
    "Create cinematic superhero VFX scenes using "
    "LangChain + Hugging Face."
)


# -----------------------------
# User Inputs
# -----------------------------

scene = st.text_area(
    "🎭 Describe your scene",
    value="""
A powerful superhero standing on top of a skyscraper
during a stormy night in a futuristic city,
wearing a futuristic armored superhero suit.
""",
    height=130
)


effects = st.text_area(
    "⚡ Describe the VFX effects",
    value="""
Intense glowing energy around both hands,
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


# -----------------------------
# LangChain Prompt Template
# -----------------------------

prompt_template = PromptTemplate.from_template("""
Create a cinematic superhero VFX scene.

Scene:
{scene}

VFX:
{effects}

Style:
epic superhero blockbuster, cinematic composition,
dramatic lighting, powerful energy effects,
glowing particles, volumetric smoke,
realistic CGI, detailed environment,
high contrast, dynamic action shot,
premium movie visual effects, photorealistic
""")


# -----------------------------
# Generate Button
# -----------------------------

if st.button("🎬 Generate VFX", type="primary"):

    if not scene.strip():
        st.warning("Please describe your scene.")
        st.stop()

    if not effects.strip():
        st.warning("Please describe the VFX effects.")
        st.stop()

    # Create prompt using LangChain
    prompt = prompt_template.format(
        scene=scene,
        effects=effects
    )

    with st.spinner("Generating your VFX scene..."):

        try:

            # -----------------------------
            # Hugging Face Client
            # -----------------------------

            hf_token = os.getenv("HF_TOKEN")

            if not hf_token:
                st.error("HF_TOKEN was not found in your .env file.")
                st.stop()

            client = InferenceClient(
                provider="fal-ai",
                api_key=hf_token
            )


            # -----------------------------
            # Generate Image
            # -----------------------------

            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )


            # -----------------------------
            # Display Image
            # -----------------------------

            st.success("VFX scene generated successfully!")

            st.image(
                image,
                caption="Generated Superhero VFX Scene",
                use_container_width=True
            )


            # -----------------------------
            # Convert Image to Bytes
            # -----------------------------

            buffer = BytesIO()

            image.save(
                buffer,
                format="PNG"
            )

            image_bytes = buffer.getvalue()


            # -----------------------------
            # Download Button
            # -----------------------------

            st.download_button(
                label="⬇️ Download Image",
                data=image_bytes,
                file_name="superhero_vfx.png",
                mime="image/png"
            )


        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )