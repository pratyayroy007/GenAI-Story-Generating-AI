import streamlit as st
from story_generator_hf import generate_story

st.set_page_config(page_title="AI Story Generator")

st.title(" AI Story Generator")
st.write("Enter a story prompt and let AI write a story for you!")

prompt = st.text_input(" Your Story Prompt", placeholder="e.g. A robot finds an ancient scroll in the desert")

length = st.slider("Story Length (in tokens)", min_value=50, max_value=300, value=500, step=10)
temperature = st.slider("Creativity (temperature)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)

if st.button("Generate Story"):
    if prompt.strip():
        with st.spinner("Generating story..."):
            story = generate_story(prompt, max_new_tokens=length, temperature=temperature)
        st.subheader(" Generated Story")
        st.text_area("Generated Story",value=story, height=300)
    else:
        st.warning("Please enter a prompt.")
