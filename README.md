# RAG를 활용한 한국사 기반 chatbot

## 가상환경 및 패키지 설치(깃 클론 하거나 파일 받고 열었는데 가상환경 셋팅 필요하면 하셈)
- 가상환경 만들기
~~~
python3 -m venv venv
~~~
- 가상환경 실행 및 종료
~~~ python
source venv/bin/activate # 실행
deactivate # 종료

~~~
- 라이브러리 설치
  - 추가 설치해야될게 있을수도 구글링이나 GPT 이용 바람 ..
    - import가 안된다면 복붙해서 구글링해보면 됨 
``` python
    pip install langchain langchain_openai langchain_community
    pip install openai faiss-cpu tiktoken cohere python-dotenv
    pip install streamlit
    pip install pymupdf4llm pymupdf
    pip install urllib3==1.26.6 # 얘는 설치하라하면 하셈
```

## 만약에 잘 안된다!
- 따로 작업환경을 만든다(vscode에 폴더 만들라느거임)
  - 가상환경 만들기(위에 설명)
    - venv 디렉토리가 생성될거임
      - 하위 폴더로 위에서 다운받은 src,data,model,test,test2폴더 넣으셈 README.md 파일도 넣기.
      - .env는 따로 만들기.(아래 말해둠.)
      - 위에 말한 패키지들 설치하면 사용가능할거임
      - 라이브러리 사용 가능해진다는 뜻.


# 안내 사항

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
- 검색 후 프로세스로 ReRank를 사용함. 
  
# 문제점
- 답변 생성까지 오래걸림
- 검색 전 프로세스를 구현해야됨. 하지만 보류 ReRank로 충분할수도 고민해봐야할듯

# 해야할 일

- 채팅 UI 수정 
- 뭔소리인지 모르겠는 코드는 물어보거나 지피티 고고!

# 12 /4일 변경 사항 
- pip install sentence-transformers 추가 설치 
-  DB 변경 chroma -> faiss
- 메타데이터는 포기함
- 테스트 파일 없앰.
- db 디렉토리 기존 venv/model -> venv/data/db
- 모든 데이터 db에 저장  했음. 
- venv/src 폴더 하위에 있는 파일들만 수정했음 위에만 업데이트해서 올려뒀음
- 데이터 파일이 너무 커서 깃허브에 안올라감 소스코드만 받고 하셈
  - split.py 파일 실행시키면 db에 저장 될거임 -> 오래걸ㄹ리니까 기다리셈
  - venv/data/db 경로에 저장될 예정이라 venv/data/ 에 db 디렉토리 생성먼저 해야댐
- 스트림릿 실행은 llm.py로 하면 됨 위에 참고
