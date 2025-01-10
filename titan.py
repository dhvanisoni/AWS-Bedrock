import json
import os
import sys
import boto3
import numpy as np
import streamlit as st

# Titan Embeddings model to generate embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
# After loading the documents we need text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Embedding and Vector Store
from langchain.vectorstores import FAISS   ## faiss cpu 

## LLM Models from Langchain 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion and load documents
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # In our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and Vector Store 
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local('faiss_index')

### Define LLM

def get_titan_llm():
    ## Create the Titan model
    llm = Bedrock(model_id="amazon.titan-text-premier-v1:0", client=bedrock, model_kwargs={'maxTokenCount':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

summary_prompt_template = """
Human: Summarize the following content in a concise manner with at least 250 words.
<context>
{context}
</context>

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
SUMMARY_PROMPT = PromptTemplate(template=summary_prompt_template, input_variables=["context"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def summarize_pdf(llm, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)
    summary = chain.run({"context": context})
    return summary

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
    
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_titan_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Summarize PDF"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_titan_llm()  # Using Titan LLM for summarization
            fixed_query = "Summarize the pdf"
            summary = summarize_pdf(llm, faiss_index, fixed_query)
            st.write(summary)
            st.success("Done")

if __name__ == "__main__":
    main()
