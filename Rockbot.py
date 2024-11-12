#!/usr/bin/env python
# coding: utf-8



# In[3]:


from pymilvus import connections

connections.connect(
    alias="default",
    host="10.10.30.6",
    port="19530"
)
print("Milvus에 연결되었습니다!")


# In[4]:


# MySQL 데이터베이스 연결 설정 함수
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='10.10.200.6',        
            port=3306,
            database='rock',
            user='rock',
            password='!!rock1234'
        )
        if connection.is_connected():
            print("데이터베이스에 연결되었습니다.")
        else:
            print("데이터베이스 연결에 실패했습니다.")
        return connection
    except Error as e:
        print(f"데이터베이스 연결 실패: {e}")
        return None


# In[5]:

def connect_milvus():
    connections.connect("default", host="10.10.30.6", port="19530")
# OpenAI API 설정
def set_openai_api():
    openai.api_key = "Your API"


# In[6]:


# 이전 대화 내용을 저장하는 변수
conversation_history = []

# 랜덤 장소 추천 키워드
random_keywords = ["한국", "대한민국", "촬영지", "촬영장소", "촬영"]


# In[7]:


# Milvus에서 저장된 region 값을 가져오는 함수
from pymilvus import connections, Collection, utility

def get_all_regions_from_milvus(collection_name="caption_collection"):
    connect_milvus()
    collection = Collection(collection_name)
    if not utility.has_collection(collection_name) or collection.is_empty:
        return []
    
    # 모든 region 값 가져오기
    query_results = collection.query(expr="region != ''", output_fields=["region"], limit=10000)
    unique_regions = list(set(result['region'] for result in query_results if result['region']))
    return unique_regions
print("regions")
all_regions = get_all_regions_from_milvus()


# In[ ]:





# In[8]:


# 필요한 라이브러리 임포트
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from pymilvus import Collection, connections, utility
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Milvus
import re
import random
import datetime

# Milvus 연결 설정 함수
def connect_milvus(host="10.10.30.6", port="19530"):
    connections.connect(alias="default", host=host, port=port)

# RAG 기반 질문 응답 함수
def answer_question_rag(question, collection_name="caption_collection", num_recommendations=5, max_tokens=200, random_mode=False):
    # Milvus 연결
    connect_milvus()

    # 컬렉션 존재 여부 확인
    if not utility.has_collection(collection_name):
        return f"컬렉션 '{collection_name}'이(가) 존재하지 않습니다."

    # 컬렉션 로드 및 인덱스 생성 확인
    collection = Collection(collection_name)
    if not collection.has_index():
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    collection.load()
    if collection.is_empty:
        return "컬렉션에 저장된 데이터가 없습니다."

    # 랜덤 추천 모드일 경우 모든 장소에서 데이터 조회
    if random_mode:
        query_results = collection.query(expr="region != ''", output_fields=["keyword", "caption"], limit=10000)
    else:
        # 질문에서 지역 추출 및 처리
        region_match = re.search(r"\b([가-힣]+(?:광역시)?)(?:에서|지역|광역시)?\b", question)  # 단순하게 한글 단어만 추출하도록 수정
        region = region_match.group(1) if region_match else None
        print(f"Extracted region: {region}")  # 디버깅 로그 추가

        query_results = []
        if region:
            if region in ["광주", "광주광역시", "광주 광역시"]:
                # 광주와 관련된 입력을 '광주(광역시)'로 변환
                region = "광주(광역시)"
                print("Converted region to 광주(광역시) for searching.")
            elif region in ["경기광주", "경기 광주", "경기도 광주"]:
                # 경기광주와 관련된 입력을 '광주(경기도)'로 변환
                region = "광주(경기도)"
                print("Converted region to 광주(경기도) for searching.")

            # 변환된 region을 기반으로 Milvus 쿼리
            query_results = collection.query(expr=f"region == '{region}'", output_fields=["keyword", "caption"], limit=10000)
        else:
            # 지역이 지정되지 않았을 경우 기본 데이터를 조회
            print("No specific region provided, querying all data.")
            query_results = collection.query(expr="region != ''", output_fields=["keyword", "caption"], limit=10000)

    # 결과를 랜덤으로 선택하여 num_recommendations만큼 제한
    if len(query_results) > num_recommendations:
        random.shuffle(query_results)
        query_results = query_results[:num_recommendations]

    if not query_results:
        # 검색 결과가 없을 때 기본 안내 메시지 반환
        return "추천할 장소가 없습니다. 더 구체적인 지역명을 입력해 주세요."

    # 검색 결과로 컨텍스트 생성
    keyword_list = [result['keyword'] for result in query_results if result['keyword']]
    caption_list = [result['caption'] for result in query_results if result['caption']]

    # LLMChain을 사용하여 각 키워드에 대한 설명 생성
    set_openai_api()
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai.api_key, max_tokens=max_tokens)
    prompt = PromptTemplate(
        input_variables=["keyword", "caption"],
        template="키워드 '{keyword}'에 대해, 다음 설명을 사용하여 간결하게 요약해 주세요: '{caption}'"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    detailed_explanations = []
    for idx, (keyword, caption) in enumerate(zip(keyword_list, caption_list), 1):
        try:
            explanation = chain.run(keyword=keyword, caption=caption)
        except Exception as e:
            explanation = f"설명을 생성할 수 없습니다. 오류: {str(e)}"
        detailed_explanations.append(f"{idx}. {keyword}: {explanation.strip()}")

    final_answer = "\n".join(detailed_explanations)
    final_answer += "\n\n출처: 한국형 데이터"

    # 이전 대화 기록에 추가
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"Bot: {final_answer}")

    return final_answer

# 체인 기반 질문 응답 함수
def answer_question_chain(question):
    openai_api_key = "your_API"
    # Milvus 컬렉션 이름 및 설정
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    milvus_connection_args = {
        "host": "10.10.30.6",
        "port": "19530"
    }

    # Milvus를 이용한 retriever 생성
    retriever = Milvus(
        collection_name="company_docs",
        embedding_function=embeddings,
        connection_args=milvus_connection_args
    ).as_retriever()

    # 프롬프트 생성
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in Korean.

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:

        
        출처: 회사 표준 규정집.pdf
        """
    )

    # 언어모델 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    # 체인 생성 및 실행
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response

def handle_general_conversation(question):
    if re.search(r"안녕|안녕하세요|반가워", question):
        response = "안녕하세요! 무엇을 도와드릴까요?"
    elif re.search(r"고마워|감사합니다|수고했어|수고하세요", question):
        response = "천만에요! 도움이 되었다니 기쁩니다. 또 궁금한 점이 있으면 언제든 물어보세요."
    else:
        # 일반적인 질문에 대해 다양한 프롬프트를 사용하여 OpenAI API로 응답 생성
        try:
            set_openai_api()
            llm = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=openai.api_key,
                max_tokens=500,
                temperature=0.6  # 온도 값을 높여 다양성을 증가시킵니다
            )

            # 다채로운 응답을 위해 여러 프롬프트 설정
            prompt_templates = [
                "질문에 대해 답변해 주세요: {question}",
                "아래 질문에 대해 자세히 설명해 주세요: {question}",
                "다음 질문에 대해 도움을 줄 수 있나요? {question}"
            ]
            selected_prompt = random.choice(prompt_templates)
            
            prompt = PromptTemplate(
                input_variables=["question"],
                template=selected_prompt
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(question=question)
        except Exception as e:
            response = f"답변을 생성할 수 없습니다. 오류: {str(e)}"

    # 이전 대화 기록에 추가
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"Bot: {response}")

    return response

# 출장 신청 데이터 저장 함수
def save_trip_application(data, user_id, department_id,name):
    print("save_trip_application 함수 시작")  # 함수 시작 확인용 메시지
    conn = get_db_connection()
    if conn is None:
        print("데이터베이스에 연결할 수 없습니다.")  # 연결 실패 메시지
        return "DB connection error"
    
    try:
        cursor = conn.cursor()
        print("DB 커서 생성 완료")  # 커서 생성 확인용 메시지
        sql = '''
            INSERT INTO travel_request (user_id, department_id, destination, travel_date, return_date, reason, status, submission_date,name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        cursor.execute(sql, (
            user_id, 
            department_id, 
            data.get("destination"), 
            data.get("travel_date"), 
            data.get("return_date"), 
            data.get("reason"), 
            'Pending',
            datetime.datetime.now(),
            name
        ))
        conn.commit()
        print("DB 커밋 완료 - 데이터 저장 성공")  # 커밋 성공 메시지
        print(f"저장된 데이터: {data}")
        return True
    except mysql.connector.Error as e:
        print(f"출장 신청 저장 중 오류 발생: {e}")
        return False
    finally:
        cursor.close()
        conn.close()
        print("DB 연결 종료")  # 연결 종료 확인 메시지

# 출장 신청 단계별 처리 함수
def handle_trip_application(question, user_id, department_id,name=None):
    global trip_application_stage, trip_application_data

    if trip_application_stage is None:
        trip_application_stage = "destination"
        return "출장 신청서를 작성합니다. 출장 목적지를 입력해 주세요."

    if trip_application_stage == "destination":
        trip_application_data["destination"] = question
        trip_application_stage = "travel_date"
        return "출장 시작 날짜를 입력해 주세요. (예: 2023-11-20)"

    elif trip_application_stage == "travel_date":
        trip_application_data["travel_date"] = question
        trip_application_stage = "return_date"
        return "출장 종료 날짜를 입력해 주세요. (예: 2023-11-25)"

    elif trip_application_stage == "return_date":
        trip_application_data["return_date"] = question
        trip_application_stage = "reason"
        return "출장 사유를 입력해 주세요."

    elif trip_application_stage == "reason":
        trip_application_data["reason"] = question
        trip_application_stage = "confirmation"
        return (f"입력하신 내용은 다음과 같습니다:\n"
                f"목적지: {trip_application_data['destination']}\n"
                f"출장 시작 날짜: {trip_application_data['travel_date']}\n"
                f"출장 종료 날짜: {trip_application_data['return_date']}\n"
                f"출장 사유: {trip_application_data['reason']}\n"
                "맞으면 '네'를, 아니면 '아니오'를 입력해 주세요.")

    elif trip_application_stage == "confirmation":
        if question.strip() == "네":
            success =  save_trip_application(trip_application_data, user_id, department_id, name)
            
            # 저장 성공 여부에 따라 응답을 분기
            if success:
                response = "출장 신청서가 저장되었습니다. 감사합니다."
            else:
                response = "출장 신청서 저장 중 오류가 발생했습니다. 다시 시도해 주세요."
            trip_application_stage = None
            trip_application_data = {}
            return response
        else:
            trip_application_stage = None
            trip_application_data = {}
            return "다시 시작합니다. 출장 목적지를 입력해 주세요."

# 사용자 질문을 처리하는 메인 함수
def handle_user_question(question, user_id=None, department_id=None,name=None):
    global trip_application_stage

    # 출장 신청 절차가 진행 중이면 해당 단계만 처리
    if trip_application_stage is not None:
        return handle_trip_application(question, user_id, department_id,name)

    # 출장 신청 관련 질문인지 확인
    if re.search(r"\b출장\b|\b출장 신청\b|\b출장신청\b", question):
        return handle_trip_application(question, user_id, department_id,name)

    # 기존 지역 관련 질문 처리 로직 (다른 코드들과 통합됨)
    if re.search(r"\b광주\b", question):
        return answer_question_rag("광주(광역시)")
    elif re.search(r"\b경기광주\b|\b경기 광주\b", question):
        return answer_question_rag("광주(경기도)")

    for region in all_regions:
        if region in question:
            return answer_question_rag(region)

    # RAG 기반 질문 처리
    if re.search(r"\b(월차|휴가|직무|근무|비품)\b.*(규정|정책|방법|어떻게|받는 법|사용법|신청|절차|조건)?", question):
        return answer_question_chain(question)

    # 나머지 경우는 Milvus의 region 목록을 확인
    if any(region in question for region in all_regions):
        return answer_question_rag(question)
    else:
        return handle_general_conversation(question)

    # 질문에 랜덤 키워드가 포함된 경우 전체 장소에서 랜덤 추천 모드로 처리
    if any(keyword in question for keyword in random_keywords):
        return answer_question_rag(question, random_mode=True)

    # 일반적인 질문 처리
    return handle_general_conversation(question)


# In[9]:


def save_chat_log(user_id, question, response):
    connection = get_db_connection()
    if connection is None:
        print("DB 연결 실패로 인해 로그를 저장할 수 없습니다.")
        return

    try:
        cursor = connection.cursor()
        print(f"저장하려는 user_id: '{user_id}'")  # 로그 출력
        query = '''
            INSERT INTO Chat_Logs (user_id, question, response) 
            VALUES (%s, %s, %s)
        '''
        # timestamp는 자동으로 설정되므로 전달하지 않음
        cursor.execute(query, (user_id, question, response))

        connection.commit()
    except mysql.connector.Error as e:
        if e.errno == 1452:  # 외래 키 제어 조건 오류 코드
            print(f"외래 키 제어 실패: user_id '{user_id}'가 user 테이블에 없습니다.")
        else:
            print(f"데이터베이스에 로그 저장 중 오류 발생: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# In[ ]:


from flask import Flask, request, jsonify
from pymilvus import connections, utility, Collection
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import openai
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import datetime
import re
import random
import logging

app = Flask(__name__)
CORS(app)

# 여행 신청 상태와 데이터를 저장할 글로벌 변수
trip_application_stage = None
trip_application_data = {}

# Flask 엔드포인트 설정
@app.route('/ask', methods=['POST'])
def ask():
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid request format. JSON data is required.'}), 400

        user_data = request.get_json()
        user_question = user_data.get('question')
        user_id = user_data.get('user_id')
        department_id = user_data.get('department_id')
        name = user_data.get('name')

        # 데이터 확인 로그 추가
        print(f"Received data: user_id={user_id}, department_id={department_id}, question={user_question}")

        if not user_question or not user_id or not department_id or not name:
            return jsonify({'error': 'user_id, department_id,name 및 질문을 제공해주세요.'}), 400

        # 질문에 대한 응답 생성
        bot_response = handle_user_question(user_question, user_id, department_id, name)

        # 채팅 로그를 DB에 저장
        save_chat_log(user_id, user_question, bot_response)

        return jsonify({'user_id': user_id, 'question': user_question, 'response': bot_response})

    except Exception as e:
        print(f"서버 오류 발생: {e}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

@app.route('/chat-logs', methods=['GET'])
def get_chat_logs():
    connection = get_db_connection()
    if connection is None:
        return jsonify({'error': 'DB 연결 실패'}), 500

    # 쿼리 파라미터에서 user_id 가져오기
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id를 제공해주세요.'}), 400

    try:
        cursor = connection.cursor(dictionary=True)
        today = datetime.datetime.now().date()
        seven_days_ago = today - datetime.timedelta(days=7)

        # 7일 이내의 로그를 날짜별로 가져오기
        query = '''
            SELECT * FROM Chat_Logs 
            WHERE user_id = %s AND DATE(timestamp) >= %s
            ORDER BY timestamp DESC
        '''
        cursor.execute(query, (user_id, seven_days_ago))
        chat_logs = cursor.fetchall()

        if not chat_logs:
            return jsonify({'today': [], 'previous_days': {}})

        # 날짜별로 로그를 그룹화
        date_grouped_logs = {}
        for log in chat_logs:
            # 로그의 timestamp에서 날짜 부분만 추출하여 문자열로 변환
            date_key = log['timestamp'].strftime('%Y-%m-%d')
            if date_key not in date_grouped_logs:
                date_grouped_logs[date_key] = []
            date_grouped_logs[date_key].append({
                'sender': 'user',
                'message': log['question'],
                'date': log['timestamp'].isoformat()
            })
            date_grouped_logs[date_key].append({
                'sender': 'bot',
                'message': log['response'],
                'date': log['timestamp'].isoformat()
            })

        # 오늘 로그와 이전 로그를 분리하여 응답 생성
        response = {
            'today': date_grouped_logs.get(today.strftime('%Y-%m-%d'), []),
            'previous_days': {
                date: logs for date, logs in date_grouped_logs.items() if date != today.strftime('%Y-%m-%d')
            }
        }

        return jsonify(response)

    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




