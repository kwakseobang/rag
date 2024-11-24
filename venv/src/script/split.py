from pdf_loader import load_data_from_pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
#  api key 로딩
load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")
# LLM이 소화할 수 있는 양으로 청크를 제한해야됨. LLM은 텍스트를 받아드릴 떄 정해진 토큰 이상으로 소화할 수 없다. 따라서 글을 토큰 단위로 분할할경우 최대한 많은 글을 포함
# 토큰 수로 하는 이유 찾기
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# dataset 파일 경로
pickle_file_path = "venv/data/processed/documents.pkl"


# Pickle 파일에서 문서 로드 (이미 로드된 데이터가 있으면 불러오기)
documents = load_data_from_pickle(pickle_file_path)

# 문자열 데이터를 Document 객체로 변환
documents = [Document(page_content=text) for text in documents]

## 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,  chunk_overlap= 100,length_function = tiktoken_len
    )
texts = text_splitter.split_documents(documents)

# 임베딩
embedding = OpenAIEmbeddings()


vectordb= FAISS.from_documents(
    documents=texts,
    embedding = embedding
).as_retriever(search_kwargs={"k": 5})

# 로컬 저장소에 저장
# 현재는 테스트에서 아래 경로에 넣음. model 디렉토리에 넣을 예정
vectordb.save_local('venv/test/')


