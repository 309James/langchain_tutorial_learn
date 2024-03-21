#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 16:36
# @Author  : james.chen
# @File    : Prompt_few_shot_eaxmple_for_chat_models.py
"""
The goal of few-shot prompt templates are to dynamically select examples based on an input,
and then format the examples in a final prompt to provide for the model.

"""

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# todo 直接使用Message无法传递input variables，弄清楚为什么。。
# example_prompt = ChatPromptTemplate.from_messages(messages=[HumanMessage(content="{input}"),
#                                                             AIMessage(content="{output}")],
#                                                   )
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
# print(example_prompt)
# few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt,
#                                                    examples=examples)
# print(few_shot_prompt.format())

# final_prompt = ChatPromptTemplate.from_messages(
#     [("system", "You are a wondrous wizard of math."),
#      few_shot_prompt,
#      ("human", "{input}"), ]
# )

# chain = final_prompt | ChatOpenAI()
# response = chain.invoke({"input": "What is the square of a triangle?"})
# print(response.content)

example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore,
                                                     k=2)

# print(example_selector.select_examples({"input": "horse"}))
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector = example_selector,
    example_prompt=example_prompt
)

print(few_shot_prompt.format(input="What's 3+3?"))

final_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a wondrous wizard of math."),
     few_shot_prompt,
     ("human", "{input}"), ]
)

chain = final_prompt | ChatOpenAI()
response = chain.invoke({"input": "What's 13+3??"})
print(response.content)