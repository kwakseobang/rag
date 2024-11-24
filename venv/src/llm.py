from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st


#  api key 로딩
load_dotenv()

# 데이터 로드
def data_load():
    embedding = OpenAIEmbeddings()
    # DB path
    DB_PATH = "venv/test/"
    vdb = FAISS.load_local(DB_PATH, embedding, allow_dangerous_deserialization=True)
    return vdb



    # prompt 생성
prompt = PromptTemplate.from_template(
    """당신은 질문에 답변하는 작업을 돕는 어시스턴트입니다.
    다음 제공된 맥락(Context)을 사용해 질문에 답하세요.
    만약에 제공된 맥락을 사용해서 질문에 답변을 못하겠다면 그냥 너가
    검색해서 알아서 답변해줘. 그 대신 맥락에서 제공 안된다고 말하지는 마.
    답변은 한국어로 작성하세요.

    #맥락(Context): 
    {context}

    #질문(Question):
    {question}

    #답변(Answer):"""
    )


def get_chain():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    vectorstore = data_load()
    retriever = vectorstore.as_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain




def main():
    vdb = data_load()
    st.set_page_config(
        page_title="HistoryChat"
    )
    st.title("챗봇 Title")

    # 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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


    st.session_state.conversation = get_chain()
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
    main()
    print()


    
    


