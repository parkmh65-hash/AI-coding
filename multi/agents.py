from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from utils import (
    get_outline,
    save_outline
)


llm = ChatOpenAI(
    model="gpt-4o"
)



def content_strategist(
    state,
    thread_id
):

    print(
        "CONTENT STRATEGIST"
    )


    prompt = PromptTemplate.from_template(
"""
너는 책 집필 AI팀의 콘텐츠 전략가이다.

사용자의 요구를 분석하여
책의 상세 목차를 작성한다.

기존 목차:
{outline}

대화:
{messages}
"""
    )


    chain = (
        prompt
        |
        llm
        |
        StrOutputParser()
    )


    result = chain.invoke(
        {
            "outline":
                get_outline(thread_id),

            "messages":
                state["messages"]
        }
    )


    save_outline(
        thread_id,
        result
    )


    state["messages"].append(
        AIMessage(
            content=
            "[Content Strategist] 목차 작성 완료"
        )
    )


    return state





def communicator(
    state
):


    prompt=PromptTemplate.from_template(
"""
너는 책 작성팀의 커뮤니케이터이다.

사용자에게 진행상황을 설명하고
다음 작업을 위한 대화를 한다.

대화:
{messages}
"""
    )


    chain=prompt | llm


    result=chain.invoke(
        {
            "messages":
            state["messages"]
        }
    )


    state["messages"].append(
        result
    )


    return state
