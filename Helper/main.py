import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

def prepare_vector_db():
    # 학습시키고 싶은 문서
    file_path = "data/fastapi_docs.txt"
    if not os.path.exists(file_path):
        content = "FastAPI는 파이썬 3.8+ 기반의 아주 빠른 웹 프레임워크입니다."
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    # 텍스트 분할
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([content])

    # 로컬 임베딩 모델 (llama3.2 사용)
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory="./chroma_db_local"
    )
    return db

# 서버 시작 시 로컬 DB 로드
vector_db = prepare_vector_db()

# 2. 로컬 LLM 설정 (Ollama)
llm = ChatOllama(model="llama3.2", temperature=0)

# 3. RAG 체인 구성 (LCEL 방식)
prompt = ChatPromptTemplate.from_template("""
당신은 백엔드 개발 전문가입니다. 아래 제공된 문서를 참고하여 한국어로 답변하세요.
문서 내용: {context}
사용자 질문: {question}
""")

# 파이프라인 구성
rag_chain = (
    {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_ai(question: Question):
    # 로컬에서 추론 실행
    response = rag_chain.invoke(question.text)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)