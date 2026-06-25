현재 작성된 코드 기준으로는 "한 질문에 3개의 도구가 동시에 실행되지는 않습니다."

동작 방식은 다음과 같습니다.

사용자 질문 수신
RAG 검색 수행
LLM이 질문과 RAG 결과를 보고 판단
필요한 도구를 선택해서 호출
도구 결과를 다시 LLM에 전달
LLM이 추가 도구가 필요하다고 판단하면 다음 도구 호출
최종 답변 생성

즉,

질문
 ↓
RAG
 ↓
LLM
 ↓
웹검색
 ↓
LLM
 ↓
유튜브검색
 ↓
LLM
 ↓
시간조회
 ↓
LLM
 ↓
최종답변

처럼 순차적으로(tool-call loop) 동작합니다.

예를 들어 사용자가

오늘 서울 시간 기준으로 AI 뉴스 알려주고 관련 유튜브 영상도 추천해줘

라고 질문하면 LLM이

get_current_time(...)

호출 →

get_web_search(...)

호출 →

get_youtube_search(...)

호출

할 수도 있습니다.

하지만 이것은

for tool_call in response.tool_calls:

구조에 의해 순서대로 실행되는 것이지,

time
web
youtube

를 무조건 모두 실행하는 구조는 아닙니다.

현재 코드의 특징

LLM이 판단

질문: 지금 몇시야?

↓

시간 도구만 사용

get_current_time()
질문: 챗GPT 최신 뉴스 알려줘

↓

웹검색만 사용

get_web_search()
질문: LangChain 강의 영상 찾아줘

↓

유튜브만 사용

get_youtube_search()

즉 필요한 도구만 선택적으로 호출합니다.

만약 항상 3개를 모두 실행하고 싶다면

LLM Agent를 사용하지 말고

time_result = get_current_time.invoke(...)
web_result = get_web_search.invoke(...)
youtube_result = get_youtube_search.invoke(...)
def run(query, history):

    docs = retriever.invoke(query)

    time_result = get_current_time.invoke({
        "timezone":"Asia/Seoul",
        "location":"서울"
    })

    web_result = get_web_search.invoke({
        "query": query,
        "search_period":"m"
    })

    youtube_result = get_youtube_search.invoke({
        "query": query
    })

    final_prompt = f"""
RAG:
{docs}

현재시간:
{time_result}

웹검색:
{web_result}

유튜브:
{youtube_result}

질문:
{query}
"""

    answer = llm.invoke(final_prompt)

    return answer

이 경우

질문
 ↓
RAG
 ↓
시간조회
 ↓
웹검색
 ↓
유튜브검색
 ↓
GPT
 ↓
답변

이 되어 매 질문마다 3개 도구가 모두 실행됩니다.

개인적으로는 지금 구조보다 병렬 실행 구조가 더 좋습니다.

예를 들면:

from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:

    future_web = executor.submit(
        get_web_search.invoke,
        {"query": query, "search_period":"m"}
    )

    future_youtube = executor.submit(
        get_youtube_search.invoke,
        {"query": query}
    )

    future_time = executor.submit(
        get_current_time.invoke,
        {
            "timezone":"Asia/Seoul",
            "location":"서울"
        }
    )

    web_result = future_web.result()
    youtube_result = future_youtube.result()
    time_result = future_time.result()

그러면 웹검색·유튜브·시간 조회가 동시에 수행되고, 마지막에 GPT가 종합해서 답변하므로 응답 속도가 훨씬 좋아집니다. 특히 Render 환경에서는 이 방식이 체감 성능이 좋습니다.

더 스마트한 응답, 파일 및 이미지 업로드 등을 이용하세요.
