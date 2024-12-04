from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import concurrent.futures

#  api key 로딩
load_dotenv()

# DB load
def db_load():
    embedding = OpenAIEmbeddings()
    DB_PATH = "venv/test/" # 해당 경로에 있는 DB를 가져오는 것임.
    retriever = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
    return retriever.as_retriever(search_kwargs={"k": 5})
# prompt 생성
prompt = PromptTemplate.from_template(
    """당신은 질문에 답변하는 작업을 돕는 어시스턴트입니다.
    다음 제공된 맥락(Context)을 사용해 질문에 답하세요.
    만약에 제공된 맥락을 사용해서 질문에 답변을 못하겠다면 그냥 너가
    검색해서 알아서 답변해줘.
    답변은 한국어로 작성하세요.

    #맥락(Context): 
    {context}

    #질문(Question):
    {question}

    #답변(Answer):"""
    )

# 냅두면 됨
def format_docs(docs):
    return

# 검색 후 프로세스 -> 리랭크
def relank_retriever():
    retriever = db_load()
    # 모델 초기화 (한국어 문서 리랭킹을 위한 사전 학습된 모델)
    model = HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker")
    # 검색된 문서 중에 상위 5개의 문서 선택
    compressor = CrossEncoderReranker(model=model, top_n=5)
    # 리랭커 기반 검색기 초기화
    relank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return relank_retriever

# chain 연결 (LLM)
def get_chain():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = relank_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# def get_answer(query):

#     # 검색 작업
#     retriever_results = relank_retriever()  # 검색 결과를 가져옵니다.
#     documents = retriever_results.invoke(query)  # 검색된 문서들
#      # 검색된 문서를 문자열로 변환 (예: 문서 내용을 텍스트 형식으로 변환)

#     # 검색된 문서에서 'page_content'를 추출하여 하나의 문자열로 변환
#     context = " ".join([doc.page_content for doc in documents]) 

#     # # 생성 작업
#     chain = st.session_state.conversation  # 세션 상태에서 기존 체인 사용

#     # 검색된 문서를 맥락으로 전달하여 답변 생성
#     result = chain.invoke({"context": context, "question": query})
    
#     return result


def streamlit_app():
    st.set_page_config(
        page_title="HistoryChat"
    )
    st.title("챗봇 Title 넣으세요")

    # 초기화
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = get_chain() 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 자랑스러운 한국사에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]


    # 메시지 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    # 사용자 입력 처리
    if query := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": query})
        # 사용자 질문 출력
        with st.chat_message("user"):
            st.markdown(query)

        # 답변 출력
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain.invoke(query)
                st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    streamlit_app()



    
    


