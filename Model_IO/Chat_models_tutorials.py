#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 22:02
# @Author  : james.chen
# @File    : Chat_models_tutorials.py
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI()


# Building functions
class Multiply(BaseModel):
    """Multiply two integers together. when the question need multiply, this tool must be used!"""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


llm_with_tools = llm.bind_functions([Multiply])
# print(llm_with_tools)
# print(llm.invoke("What's 3*12?").content)
r = llm_with_tools.invoke("What's 7*12?")
print(r)

# outparser
from langchain_core.output_parsers import JsonOutputParser

chain = llm_with_tools|JsonOutputParser()
print(chain.invoke("What's 12*9"))

