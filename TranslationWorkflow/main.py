import os
from dotenv import load_dotenv
from prompts import *
load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from typing import TypedDict, Literal, Optional
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model=os.getenv("MODEL"),
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

class GraphState(TypedDict):
    original_text: str
    source_language: str
    target_language: str
    initial_translation: str
    quality_check: Literal["Good", "Bad"]
    reflection: Optional[str]
    final_translation: str

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

translate_prompt = PromptTemplate(
    template= TRANSLATE_TEMPLATE,
    input_variables=["source_language", "target_language", "text"]
)
translation_chain = translate_prompt | model | StrOutputParser()
def translate(state: GraphState):
    """
    接受原始文本，进行第一次翻译
    """
    initial_translation = translation_chain.invoke({
        "source_language": state["source_language"],
        "target_language": state["target_language"],
        "text": state["original_text"]
    })
    return {"initial_translation": initial_translation}


quality_check_prompt = PromptTemplate(
    template=QUALITY_CHECK_PROMPT,
    input_variables=["source_language", "target_language", "original_text", "initial_translation"]
)

quality_check_chain = quality_check_prompt | model | StrOutputParser()

def check_quality(state: GraphState) -> Literal["continue_to_refine", "end_process"]:
    """
    检查初版翻译的质量，并决定后续操作
    """
    quality_assessment = quality_check_chain.invoke({
        "source_language": state["source_language"],
        "target_language": state["target_language"],
        "original_text": state["original_text"],
        "initial_translation": state["initial_translation"],
    })
    if "Good" in quality_assessment:
        return "end_process"
    return "continue_to_refine"


def finalize_translation(state: GraphState):
    """初版翻译质量足够好，直接作为最终结果"""
    return {"final_translation": state["initial_translation"]}


reflection_prompt = PromptTemplate(
    template= REFLECTION_TEMPLATE,
    input_variables=["original_text", "initial_translation", "source_language", "target_language"],
)

reflection_chain = reflection_prompt | model | StrOutputParser()

def reflect(state: GraphState):
    """
    审核第一次的翻译结果，并提出修改建议
    """
    reflection_text = reflection_chain.invoke({
        "source_language": state["source_language"],
        "target_language": state["target_language"],
        "original_text": state["original_text"],
        "initial_translation": state["initial_translation"],
    })
    return {"reflection": reflection_text}



refine_prompt = PromptTemplate(
    template=REFINE_TEMPLATE,
    input_variables=["original_text", "initial_translation", "reflection", "source_language", "target_language"]
)

refine_chain = refine_prompt | model | StrOutputParser()

def refine(state: GraphState):
    """
    结合原始文本，初版翻译和反思建议，生成优化版翻译
    """
    final_translation = refine_chain.invoke({
        "source_language": state["source_language"],
        "target_language": state["target_language"],
        "original_text": state["original_text"],
        "initial_translation": state["initial_translation"],
        "reflection": state["reflection"],
    })
    return {"final_translation": final_translation}


from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)
workflow.add_node("translate", translate)
workflow.add_node("reflect", reflect)
workflow.add_node("refine", refine)
workflow.add_node("finalize_translation", finalize_translation)

workflow.add_edge(START, "translate")
workflow.add_conditional_edges(
    "translate",
    check_quality,
    {
        "continue_to_refine": "reflect",
        "end_process": "finalize_translation"
    }
)
workflow.add_edge("reflect", "refine")
workflow.add_edge("refine", END)
workflow.add_edge("finalize_translation", END)

graph = workflow.compile()


def translate_website(url: str):
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"Start loading content from: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content_to_translate = docs[0].page_content
    except Exception as e:
        print(f"Error loading website: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n", "\n\n"]
    )
    chunks = text_splitter.split_text(content_to_translate)

    final_translated_text = []

    for i, chunk in enumerate(chunks):
        print(f"===== Processing Chunk {i+1}/{len(chunks)} =====")
        inputs = {
            "original_text": chunk,
            "source_language": "English",
            "target_language": "Simplified Chinese"
        }
        result_state = graph.invoke(inputs)

        # print("--- 原始文本 ---")
        # print(result_state.get('original_text', 'N/A'))
        #
        # print("--- 初版翻译 ---")
        # print(result_state.get('initial_translation', 'N/A'))
        #
        # if result_state.get('reflection'):
        #     print("--- 反思与建议 ---")
        #     print(result_state['reflection'])
        #     print("--- [优化后] 最终翻译 ---")
        # else:
        #     print("--- 最终翻译 ---")

        final_text = result_state.get('final_translation')
        print(final_text)
        if final_text:
            final_translated_text.append(final_text)

    print("\n" + "=" * 30)
    print("FULL TRANSLATION RESULT")
    print("=" * 30)
    print("\n".join(final_translated_text))


if __name__ == "__main__":
    web_url = "https://www.ruanyifeng.com/calvino/2007/09/winter_s_night_ch_1_en.html"
    translate_website(web_url)