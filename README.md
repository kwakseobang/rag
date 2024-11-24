# RAG를 활용한 한국사 기반 chatbot

## 가상환경 및 패키지 설치(깃 클론 하거나 파일 받고 열었는데 가상환경 셋팅 필요하면 하셈)
- 가상환경 만들기
~~~
python3 -m venv venv
~~~
- 가상환경 실행
~~~
source venv/bin/activate
~~~
- 라이브러리 설치
  - 추가 설치해야될게 있을수도 구글링이나 GPT 이용 바람 ..
``` python
    pip install langchain langchain_openai langchain_community
    pip install openai faiss-cpu tiktoken cohere python-dotenv
    pip install streamlit
    pip install pymupdf4llm pymupdf
    pip install urllib3==1.26.6 #얘는 설치하라하면 하셈
```


# 안내 사항
- 현재 src 디렉토리 안에 test파일과 test_llm으로 작업중임.
- pdf_loder 파일만 test 파일 따로 안 만들고 사용중임 안에 경로만 바꿔서 사용중임 보면 알 듯. 한 디렉토리만 바꿔서 사용중임
  
- data/prcessed 안에 파일 저장해둔건 테스트할때마다 문서들을 다운받을 수 없으니 로컬에 저장해둔거임
- .env 파일 만들고 API Key 넣으면 됨 유출되면 안됨(파일 보내드림)
- 나머지는 주석 보셈
- 
# 스트림릿 실행 방법 
  - vs 터미널에서 아래 입력
~~~ python
    streamlit run 파일경로
    # 예시 ( 현재 파일 위치 final_project/ )
    # streamlit run venv/scr/test_llm.py
~~~
  - 종료 : 컨트롤 + c

# 진행 사항
- 전처리를 진행했으나 수정이 필요함 내가 계속 하겠음
- 백터 db에 저장할때 각 청킹에 메타데이터를 추가함 
  - 메타데이터로 파일 이름과 경로를 추가함.
- 검색 후 프로세스로 ReRank를 사용함. 
- 현재 어느 정도 챗봇도 구현되게 함
  - 하지만 이전 내용을 기억하진 못함. 이전 내용도 기억하게 해야됨
  - ex) 질문 1 날리고 질문 2날릴 때 질문 1 내용을 기억 못함
  
# 문제점
- 답변 생성까지 오래걸림
- 검색 전 프로세스를 구현해야됨. 하지만 보류 ReRank로 충분할수도 고민해봐야할듯
- 데이터 인덱싱 최적화 필요.
- 성능 비교 및 검증 해야됨
# 해야할 일
- 대화 내용을 기억하게 해주면 될 듯 
- 채팅 UI 수정 
- 뭔소리인지 모르겠는 코드는 물어보거나 지피티 고고!