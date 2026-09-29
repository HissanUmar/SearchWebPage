import streamlit as st
from pageqa.pipeline import run

st.title("Page Q&A")
colab = st.sidebar.text_input("Colab ngrok URL", placeholder="https://xxxx.ngrok-free.app")
url = st.text_input("Page URL")
question = st.text_input("Question")

if st.button("Ask"):
    if not (colab and url and question):
        st.warning("Fill in the Colab URL, page URL and question.")
        st.stop()
    with st.spinner("Working..."):
        try:
            a = run(url.strip(), question.strip(), colab.strip())
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    if a.purpose:
        st.caption(f"Page purpose: {a.purpose}")
    if not a.found:
        st.warning("Not found on this page.")
        st.stop()

    st.markdown(a.explanation)
    for s in a.sources:
        st.subheader(s.chunk.path)
        st.markdown(f"[Open section]({s.link})")
        if s.chunk.text:
            st.markdown(s.chunk.text)
        for code in s.chunk.code:
            st.code(code)
