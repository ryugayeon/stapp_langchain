#기본 정보 입력
import streamlit as st
from PyPDF2 import PdfReader

#langchain 패키지들
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from googletrans import Translator

#기능 구현 함수
def google_trans(messages):
    google = Translator()
    result = google.translate(messages, dest='ko')
    return result.text

#메인 함수
def main():
    st.set_page_config(page_title='PDF analyzer', layout='wide')

    #사이드바
    with st.sidebar:
        open_apikey = st.text_input(label='🔑OPEN API 키🖌️', placeholder='Enter Your API Key', value='', type = 'password')
        
        #입력받은 API 키 표시
        if open_apikey:
            st.session_state['OPENAI_API'] = open_apikey
        st.markdown('---')    
    
    #메인
    st.header(" PDF 내용 질문 프로그램")
    st.markdown('---')
    st.subheader('🔹 PDF 파일을 넣으세요')

    #PDF 파일 받기
    pdf = st.file_uploader(" ", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)    
        st.markdown('---')
        st.subheader("🔹 질문을 입력하세요.")
        
        # 사용자 질문 받기
        user_question = st.text_input("❓Ask a question about PDF:")
        if user_question:
            #임베딩, 시멘팅 인덱스
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["OPENAI_API"])
            
            
            #질문하기
            #llm 모델 설정
            llm = ChatOpenAI(temperature=0,
                    openai_api_key=st.session_state["OPENAI_API"],
                    max_tokens=3000,
                    model_name='gpt-3.5-turbo',
                    request_timeout=120
                    )
            
            #랭체인을 활용한 임베딩 모델 개선    
            db = Chroma.from_texts(chunks, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
            
            qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            query = user_question
            result = qa({"query": query})
            
            #답변 결과
            st.info(result["result"])
            
            if st.button(label='번역하기'):
                trans = google_trans(result["result"])
                st.success(trans)
                
if __name__=="__main__":
    main()
        
            