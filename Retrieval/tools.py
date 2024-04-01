#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 22:08
# @Author  : james.chen
# @File    : tools.py
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_extraction_chain
from langchain_community.document_loaders import AsyncChromiumLoader, PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import find_dotenv, load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0, model="gpt-4-0613")


# 网页搜索
def WebSearch(question):
    search = BingSearchAPIWrapper()
    result_set = search.results(question, 2)
    return [result['link'] for result in result_set]


# 网页总结
def WebExtractAndSummarize(websites):
    summary = []
    loader = AsyncChromiumLoader(websites)
    html = loader.load()
    # html_2_text = Html2TextTransformer()
    # docs = html_2_text.transform_documents(html)
    bs_transformers = BeautifulSoupTransformer()
    docs_transformed = bs_transformers.transform_documents(html, tags_to_extract=['span'])

    schema = {
        "properties": {
            "news_article_title": {"type": "string"},
            "news_article_summary": {"type": "string"},
        },
        "required": ["news_article_title", "news_article_summary"],
    }

    def extract(content: str, schema: dict):
        chain = create_extraction_chain(schema, llm)
        return chain.invoke(content)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    extract_test1 = extract(splits[0].page_content, schema)
    summary = extract()
    return summary


# 文档切割
def fileSplitter(file_path):
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages

# 文档灌库
def knowledgeToLib(pages):
    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(pages, embedding_func, persist_directory="./vectorstores/gplan2.4.0")


def getLocalStore(db_path):
    return Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())



def QARetrieval(database, question):
    search_results = database.similarity_search(question, k=5)
    return [r.page_content for r in search_results]



if __name__ == '__main__':

    db_path = "./vectorstores/"
    vectorstore = getLocalStore(db_path)
    llm = ChatOpenAI(temperature=0)
    prompt = load_prompt("./prompts/GPlanQA.yaml")

    while True:

        # question = "今天天气怎么样？"
        question = input("请输入你的问题：")
        search_results = QARetrieval(vectorstore, question)
        messages = [SystemMessage(prompt.format(question=question, retrieval_content=str(search_results))),
                   HumanMessage(content=question)]
        print("messages:", messages)
        response = llm.invoke(messages)
        print("answer:")
        print(response.content)


