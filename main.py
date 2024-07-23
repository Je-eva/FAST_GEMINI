import streamlit as st
import requests

API_URL = "http://localhost:8000"

def upload_pdf(pdf_files):
    files = [("pdf_docs", (pdf.name, pdf, "application/pdf")) for pdf in pdf_files]
    response = requests.post(f"{API_URL}/upload-pdf/", files=files)
    return response.json()

def ask_question(question):
    data = {"question": question}
    response = requests.post(f"{API_URL}/ask", json=data)
    return response.json()

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = ask_question(user_question)
        st.write("Reply:", response.get("answer", "No response available"))

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    response = upload_pdf(pdf_docs)
                    st.success(response.get("message", "Processing done"))

if __name__ == "__main__":
    main()
