import pymupdf4llm
import os
import pickle


# 주어진 디렉토리 경로
directory_path = 'venv/data/raw'
directories = ["고대","고려시대","근대","조선시대","현대사"]


# Pickle 파일에 데이터를 저장하는 함수
def save_data_to_pickle(data, pickle_file):
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {pickle_file}")
    except Exception as e:
        print(f"Error saving data to {pickle_file}: {e}")

# Pickle 파일에서 데이터를 불러오는 함수
def load_data_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {pickle_file}")
        return data
    except Exception as e:
        print(f"Error loading data from {pickle_file}: {e}")
        return None

# 모든 PDF 파일을 마크다운 형식으로 로드하는 함수
def load_pdfs_from_directory(directory_path, directories):
    documents = []

    # 각 디렉토리마다 순차적으로 처리
    for dir in directories:
        subdir_path = os.path.join(directory_path, dir)  # 각 하위 디렉토리 경로 결합
        
        # os.walk()를 사용하여 하위 디렉토리까지 재귀적으로 탐색
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                if file.endswith(".pdf"):  # PDF 파일만 처리
                    file_path = os.path.join(root, file)  # 파일 경로 결합
                    try:
                            
                            # 페이지 텍스트를 마크다운 형식으로 변환
                        markdown_text = pymupdf4llm.to_markdown(file_path)
                            
                            # 메타데이터에 페이지 정보 추가
                        metadata = {
                                "filename": file,
                                "directory": dir
                            }
                        # 마크다운 텍스트와 메타데이터를 함께 저장
                        document = {
                            "content": markdown_text,  # 마크다운 텍스트
                            "metadata": metadata       # 메타데이터
                        }
                        documents.append(document)  # 변환된 마크다운 텍스트를 리스트에 추가
                        print(f"Loaded markdown from {file_path}")
                    except Exception as e:
                        print(f"Error loading PDF from {file_path}: {e}")

    return documents

if __name__ == "__main__":
    # Pickle 파일 경로 
    pickle_file_path = "venv/data/processed/documents.pkl"


    # 문서 로드
    documents = load_pdfs_from_directory(directory_path, directory_path)

    # 테스트를 위해 로컬에 따로 저장 -> 문서 로드 시간을 줄이기 위해서
    save_data_to_pickle(documents, pickle_file_path)

    # 로드된 문서 수 출력
    print(f"Total documents loaded: {len(documents)}")




