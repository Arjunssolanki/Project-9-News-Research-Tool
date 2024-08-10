import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="News Research Tool", layout="wide")

st.markdown("""
## News Research Tool: Get instant insights from news articles

This tool is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It fetches the content from a news article URL, processes it to create a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the tool:

1. **Enter Your API Key**: You'll need a Google API key for the tool to access Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).

2. **Enter News URL**: Provide the URL of a news article to fetch and analyze its content.

3. **Ask a Question**: After processing the article, ask any question related to the content for a precise answer.
""")

# API Key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def fetch_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([para.get_text() for para in paragraphs])
        return content if content else None
    except requests.RequestException as e:
        st.error(f"Failed to fetch article content: {e}")
        return None

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("News Research Assistant")

    news_url = st.text_input("Enter a News Article URL", key="news_url")
    user_question = st.text_input("Ask a Question related to the News Article", key="user_question")

    if st.button("Fetch and Process News Article", key="fetch_button"):
        if api_key and news_url:
            with st.spinner("Fetching and processing article..."):
                try:
                    content = fetch_article_content(news_url)
                    if content:
                        text_chunks = get_text_chunks(content)
                        get_vector_store(text_chunks, api_key)
                        st.success("Article processed successfully!")
                    else:
                        st.warning("No content found in the provided URL.")
                except Exception as e:
                    st.error(f"Error fetching or processing article: {e}")
        else:
            st.error("Please enter your Google API Key and a valid news article URL.")

    if user_question and api_key:
        try:
            with st.spinner("Retrieving answer..."):
                user_input(user_question, api_key)
        except Exception as e:
            st.error(f"Error processing your request: {e}")

if __name__ == "__main__":
    main()
