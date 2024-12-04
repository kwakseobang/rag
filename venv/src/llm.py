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
import json
#  api key 로딩
load_dotenv()
    
@st.cache_resource
# 데이터 로드
def data_load():
    embedding = OpenAIEmbeddings()
    # DB path
    DB_PATH = "venv/data/db"
    vdb = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
    return vdb.as_retriever(search_kwargs={"k": 5})

    # prompt 생성
prompt = PromptTemplate.from_template(
    """당신은 질문에 답변하는 작업을 돕는 한국사 전문 어시스턴트입니다.

    # 이전 대화 내용:
    {chat_history}

    이전 대화를 참고하여, 다음 제공된 맥락(Context)을 사용해 질문에 답하세요.
    이전 대화에서 나온 내용이 있다면 그것을 참고하여 답변을 연결해주세요.

    만약 제공된 정보만으로 답변이 불가능하다면 추가적인 정보를 검색해서 답변하세요.
    맥락에 없을경우 맥락에 없다는 말은 하지말고 자연스럽 답변해주시는데  한국어로 작성하세요.

    # 맥락(Context): 
    {context}

    # 현재 질문(Question):
    {question}

    # 답변(Answer):"""
    )

def format_chat_history(chat_history):
    if not chat_history:
        return "이전 대화 없음"
    
    # 최근 5개의 대화만 포함
    recent_history = chat_history[-10:]  # 최대 5개의 대화 쌍(10개의 메시지)
    formatted_history = ""
    for msg in recent_history:
        speaker = "사용자" if msg["role"] == "user" else "어시스턴트"
        formatted_history += f"{speaker}: {msg['content']}\n"
    return formatted_history.strip()


def get_chat_history(_arg = None):
        if 'chat_history' in st.session_state:
            chat_history = st.session_state.chat_history
        else:
            chat_history = []
        return chat_history

def get_chain():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    vectorstore = data_load()
    retriever = relank_retriever(vectorstore)
    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": get_chat_history}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain



# 검색 후 프로세스 -> 리랭크
def relank_retriever(vdb):
    retriever = vdb
    # 모델 초기화 (한국어 문서 리랭킹을 위한 사전 학습된 모델)
    model = HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker")
    # 검색된 문서 중에 상위 5개의 문서 선택
    compressor = CrossEncoderReranker(model=model, top_n=3)
    # 리랭커 기반 검색기 초기화
    relank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return relank_retriever

# streamlit 실행
def main():
    st.set_page_config(
        page_title="HistoryChat"
    )
    st.title("챗봇 Title")

  # 초기화
    if "conversation" not in st.session_state or st.session_state.conversation is None:

        st.session_state.conversation = get_chain() 

    if "chat_history" not in st.session_state:
       st.session_state.chat_history = [] # 대화 히스토리

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
        st.session_state.chat_history.append(f"User: {query}")  # 대화 기록에 추가
        print(st.session_state.chat_history)
        # 사용자 질문 출력
        with st.chat_message("user"):
            st.markdown(query)


        # 답변 출력
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
              
                result = chain.invoke(query)  # 직접 인자로 전달
                st.markdown(result)
                st.session_state.chat_history.append( f"Assistant: {result}")  # 대화 기록에 추가
                print("답변 ",st.session_state.chat_history)

                

        st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()



    
    


