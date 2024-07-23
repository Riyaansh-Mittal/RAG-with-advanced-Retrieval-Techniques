# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.storage import InMemoryStore
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.vectorstores import Chroma
# from langchain.document_transformers import EmbeddingsRedundantFilter
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.retrievers.document_compressors import DocumentCompressorPipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import List, Union, Dict
# import os
# from dotenv import load_dotenv

# # Load the API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function for reading PDF files
# def read_pdf(pdf):
#     text = ""
#     for file in pdf:
#         pdf_read = PdfReader(file)
#         for page in pdf_read.pages:
#             text += page.extract_text()
#     return text

# # Document chunking
# def get_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Create Embeddings Store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Create Conversation Chain
# def get_conversation_chain_pdf():
#     prompt_template = """
#     Your role is to be a meticulous researcher. Answer the question using only the information found within the context.
#     Be detailed, but avoid unnecessary rambling.
#     if you cannot find the answer, simply state 'answer is not available in the context'.
#     Context: \n{context}?\n
#     Question: \n{question}?\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Create Complete Object for MultiQuery Retrieval
# class LineList(BaseModel):
#     lines: List[str] = Field(description="Lines of text")

# def create_complete_object(raw_lines: str, question: str) -> Dict[str, Union[str, LineList]]:
#     lines = [line.strip() for line in raw_lines.strip().split('\n') if line.strip()]
#     line_list = LineList(lines=lines)
#     return {'question': question, 'text': line_list}

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from a vector
#     database. By generating multiple perspectives on the user question, your goal is to help
#     the user overcome some of the limitations of the distance-based similarity search.
#     Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# # Processing User Input
# def user_input(user_query, retrieval_method):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     load_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     if retrieval_method == "ParentDocumentRetriever":
#         child_splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=20)
#         parent_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=20)
#         vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
#         store = InMemoryStore()
#         retriever = ParentDocumentRetriever(
#             vectorstore=vectorstore,
#             docstore=store,
#             child_splitter=child_splitter,
#             parent_splitter=parent_splitter
#         )
#         retriever.add_documents(load_vector_db.similarity_search(user_query))
#         docs = retriever.get_relevant_documents(user_query)
#     elif retrieval_method == "MultiQueryRetriever":
#         llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=load_vector_db.as_retriever(), llm=llm
#         )
#         unique_docs = retriever.get_relevant_documents(user_query)
#         llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)
#         response = llm_chain.run(question=user_query)
#         complete_object = create_complete_object(response, user_query)
#         docs = unique_docs
#     elif retrieval_method == "Contextual Compression":
#         vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
#         retriever = vectorstore.as_retriever()
#         retriever.get_relevant_documents(user_query)
#         splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separator=". ")
#         redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
#         relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)
#         pipeline_compressor = DocumentCompressorPipeline(
#             transformers=[splitter, redundant_filter, relevant_filter]
#         )
#         compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
#         compressed_docs = compression_retriever.get_relevant_documents(user_query)
#         docs = compressed_docs

#     chain = get_conversation_chain_pdf()
#     response = chain(
#         {"input_documents": docs, "question": user_query},
#         return_only_outputs=True
#     )
#     st.write("AI_Response", response["output_text"])

# def main():
#     st.header("Chat with your PDF files using Google Gemini Pro")
#     user_query = st.text_input("Ask a question about the PDF file?")
#     retrieval_method = st.selectbox("Choose Retrieval Method", ["ParentDocumentRetriever", "MultiQueryRetriever", "Contextual Compression"])
#     if user_query:
#         user_input(user_query, retrieval_method)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload your PDF File, and click Submit!", accept_multiple_files=True)
#         if st.button("Submit!"):
#             with st.spinner('Processing...'):
#                 raw_text = read_pdf(pdf_docs)
#                 print(raw_text)
#                 text_chunks = get_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Processing Done!")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# from PyPDF2 import PdfReader
# import spacy
# from spacy.cli import download as spacy_download
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import LLMChain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.vectorstores import Chroma
# from langchain.document_transformers import EmbeddingsRedundantFilter
# from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.storage import InMemoryStore
# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import List, Union, Dict
# import os
# from dotenv import load_dotenv

# # Ensure SpaCy model is downloaded
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     spacy_download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# # Load the API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function for reading PDF files
# def read_pdf(pdf_files):
#     text = ""
#     for file in pdf_files:
#         pdf_reader = PdfReader(file)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Preprocess and segment text into sentences
# def preprocess_text(text):
#     text = " ".join(text.split())
#     doc = nlp(text)
#     sentences = [sent.text for sent in doc.sents]
#     return sentences

# # Chunk text into segments of approximately 512 tokens with 50-token overlap
# def chunk_text(sentences, max_tokens=512, overlap=50):
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for sentence in sentences:
#         tokens = len(sentence.split())
#         if current_length + tokens > max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap:]
#             current_length = sum(len(sent.split()) for sent in current_chunk)
        
#         current_chunk.append(sentence)
#         current_length += tokens
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# # Document chunking using the preprocessing and chunking strategy
# def get_chunks(text):
#     sentences = preprocess_text(text)
#     chunks = chunk_text(sentences)
#     return chunks

# # Create Embeddings Store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Create Conversation Chain
# def get_conversation_chain_pdf():
#     prompt_template = """
#     Your role is to be a meticulous researcher. Answer the question using only the information found within the context.
#     Be detailed, but avoid unnecessary rambling.
#     If you cannot find the answer, simply state 'answer is not available in the context'.
#     Context: \n{context}?\n
#     Question: \n{question}?\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Create Complete Object for MultiQuery Retrieval
# class LineList(BaseModel):
#     lines: List[str] = Field(description="Lines of text")

# def create_complete_object(raw_lines: str, question: str) -> Dict[str, Union[str, LineList]]:
#     lines = [line.strip() for line in raw_lines.strip().split('\n') if line.strip()]
#     line_list = LineList(lines=lines)
#     return {'question': question, 'text': line_list}

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from a vector
#     database. By generating multiple perspectives on the user question, your goal is to help
#     the user overcome some of the limitations of the distance-based similarity search.
#     Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# # Processing User Input
# def user_input(user_query, retrieval_method):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     load_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     if retrieval_method == "ParentDocumentRetriever":
#         child_splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=20)
#         parent_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=20)
#         vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
#         store = InMemoryStore()
#         retriever = ParentDocumentRetriever(
#             vectorstore=vectorstore,
#             docstore=store,
#             child_splitter=child_splitter,
#             parent_splitter=parent_splitter
#         )
#         retriever.add_documents(load_vector_db.similarity_search(user_query))
#         docs = retriever.get_relevant_documents(user_query)
#     elif retrieval_method == "MultiQueryRetriever":
#         llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=load_vector_db.as_retriever(), llm=llm
#         )
#         unique_docs = retriever.get_relevant_documents(user_query)
#         llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)
#         response = llm_chain.run(question=user_query)
#         complete_object = create_complete_object(response, user_query)
#         docs = unique_docs
#     elif retrieval_method == "Contextual Compression":
#         vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
#         retriever = vectorstore.as_retriever()
#         retriever.get_relevant_documents(user_query)
#         splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separator=". ")
#         redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
#         relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)
#         pipeline_compressor = DocumentCompressorPipeline(
#             transformers=[splitter, redundant_filter, relevant_filter]
#         )
#         compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
#         compressed_docs = compression_retriever.get_relevant_documents(user_query)
#         docs = compressed_docs

#     chain = get_conversation_chain_pdf()
#     response = chain(
#         {"input_documents": docs, "question": user_query},
#         return_only_outputs=True
#     )
#     st.write("AI_Response", response["output_text"])

# def main():
#     st.header("Chat with your PDF files using Google Gemini Pro")
#     user_query = st.text_input("Ask a question about the PDF file?")
#     retrieval_method = st.selectbox("Choose Retrieval Method", ["ParentDocumentRetriever", "MultiQueryRetriever", "Contextual Compression"])
#     if user_query:
#         user_input(user_query, retrieval_method)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload your PDF File, and click Submit!", accept_multiple_files=True)
#         if st.button("Submit!"):
#             with st.spinner('Processing...'):
#                 raw_text = read_pdf(pdf_docs)
#                 text_chunks = get_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Processing Done!")

# if __name__ == "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
import spacy
from spacy.cli import download as spacy_download
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.storage import InMemoryStore
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Union, Dict
import os
from dotenv import load_dotenv
import re

# Ensure SpaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function for reading PDF files
def read_pdf(pdf_files):
    text = ""
    for file in pdf_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Clean and preprocess text
def clean_text(text):
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove references like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    return text

# Preprocess and segment text into sentences
def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Chunk text into segments of approximately 512 tokens with 50-token overlap
def chunk_text(sentences, max_tokens=512, overlap=50):
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokens = len(sentence.split())
        if current_length + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(sent.split()) for sent in current_chunk)
        
        current_chunk.append(sentence)
        current_length += tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Document chunking using the preprocessing and chunking strategy
def get_chunks(text):
    sentences = preprocess_text(text)
    chunks = chunk_text(sentences)
    return chunks

# Create Embeddings Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create Conversation Chain
def get_conversation_chain_pdf():
    prompt_template = """
    Your role is to be a meticulous researcher. Answer the question using only the information found within the context.
    Be detailed, but avoid unnecessary rambling.
    If you cannot find the answer, simply state 'answer is not available in the context'.
    Context: \n{context}?\n
    Question: \n{question}?\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Create Complete Object for MultiQuery Retrieval
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

def create_complete_object(raw_lines: str, question: str) -> Dict[str, Union[str, LineList]]:
    lines = [line.strip() for line in raw_lines.strip().split('\n') if line.strip()]
    line_list = LineList(lines=lines)
    return {'question': question, 'text': line_list}

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five to ten
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Processing User Input
def user_input(user_query, retrieval_method):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    load_vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    if retrieval_method == "ParentDocumentRetriever":
        child_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        retriever.add_documents(load_vector_db.similarity_search(user_query))
        docs = retriever.get_relevant_documents(user_query)
        
    elif retrieval_method == "MultiQueryRetriever":
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
        retriever = MultiQueryRetriever.from_llm(
            retriever=load_vector_db.as_retriever(), llm=llm
        )
        unique_docs = retriever.get_relevant_documents(user_query)
        llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)
        response = llm_chain.run(question=user_query)
        complete_object = create_complete_object(response, user_query)
        docs = unique_docs
        
    elif retrieval_method == "Contextual Compression":
        retriever = load_vector_db.as_retriever()
        splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.4)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
        compressed_docs = compression_retriever.get_relevant_documents(user_query)
        docs = compressed_docs

    chain = get_conversation_chain_pdf()
    response = chain(
        {"input_documents": docs, "question": user_query},
        return_only_outputs=True
    )
    st.write("AI_Response", response["output_text"])


def main():
    st.header("Chat with your PDF files using Google Gemini Pro")
    user_query = st.text_input("Ask a question about the PDF file?")
    retrieval_method = st.selectbox("Choose Retrieval Method", ["ParentDocumentRetriever", "MultiQueryRetriever", "Contextual Compression"])
    if user_query:
        user_input(user_query, retrieval_method)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF File, and click Submit!", accept_multiple_files=True)
        if st.button("Submit!"):
            with st.spinner('Processing...'):
                raw_text = read_pdf(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Done!")

if __name__ == "__main__":
    main()



