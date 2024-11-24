from pdf_loader import load_data_from_pickle
from pdf_loader import load_data_from_pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

#  api key 로딩
load_dotenv()

# LLM이 소화할 수 있는 양으로 청크를 제한해야됨. LLM은 텍스트를 받아드릴 떄 정해진 토큰 이상으로 소화할 수 없다. 따라서 글을 토큰 단위로 분할할경우 최대한 많은 글을 포함
tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# 문서 출력
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

# test dataset 파일 경로 -> 조선시대 후기 
test_pickle_file_path = "venv/data/processed/test_documents.pkl"


# Pickle 파일에서 문서 로드 (이미 로드된 데이터가 있으면 불러오기)
def load():
    documents = load_data_from_pickle(test_pickle_file_path)


    # # 문자열 데이터를 Document 객체로 변환
    documents = [
        Document(page_content=item['content'], metadata=item['metadata']) for item in documents
    ]
    return documents

# doc 분할
def doc_split():
    documents = load()
    # 문서 분할 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,  chunk_overlap= 100,length_function = tiktoken_len
        )
    #  문서 분할
    texts = text_splitter.split_documents(documents)
    return texts

def retriver():

    texts = doc_split()

    # 백터 스토어 저장
    retriever= FAISS.from_documents(
        documents=texts,
        embedding = OpenAIEmbeddings()
    )


   

     # 로컬 저장소에 저장
    retriever.save_local('venv/test/')

if __name__ == "__main__":
    retriver()
    


