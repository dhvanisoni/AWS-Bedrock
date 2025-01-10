# importing libraries
import json
import os
import sys
import boto3
import numpy as np
import streamlit as st

# we will be using Titan Embeddings model to generate embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
# after loading the documents we need text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Embedding and Vector Store
from langchain.vectorstores import FAISS  

## LLM Models from Langchain 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion and load documents

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    try:
        documents = loader.load()
        if not documents:
            raise ValueError("No PDF files were found in the 'data' directory.")
        print(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        raise RuntimeError(f"Error loading documents: {e}")
    
    # Split documents into chunks for processing
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        if not docs:
            raise ValueError("Text splitting returned no chunks. Check the splitter configuration.")
        print(f"Generated {len(docs)} document chunks.")
    except Exception as e:
        raise RuntimeError(f"Error splitting documents: {e}")
    
    return docs

## Vector Embedding and Vector Store 
def get_vector_store(docs):
    if not docs:
        raise ValueError("No documents provided for embedding. Please check the ingestion process.")
    
    try:
        print("Generating embeddings for documents...")
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local('faiss_index')
        print("Vector store successfully created and saved locally.")
    except Exception as e:
        raise RuntimeError(f"Error during vector store creation: {e}")

## Define LLMs
def get_Jurassic_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm

def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':512})
    return llm
    
# # anthropic.claude-3-sonnet-20240229-v1:0
# # max_tokens = 2000

def get_titan_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="amazon.titan-text-premier-v1:0",client=bedrock,
                model_kwargs={'maxTokenCount':512})
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",
                client=bedrock,
                model_kwargs={'max_gen_len':512})
    return llm

def get_mistral_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,
                model_kwargs={'max_tokens':512})
    return llm

# mistral.mistral-7b-instruct-v0:2
# max_tokens\":1000,

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explainations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template= prompt_template, input_variables=["context", "question"])

# get response from llm 
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


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

    if st.button("Jurassic Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_Jurassic_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)

            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
    
    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_mistral_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_titan_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()


















 
