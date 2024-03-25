#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 18:03
# @Author  : james.chen
# @File    : Prompt_pipeline.py
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv())
llm = ChatOpenAI()
# print(llm.invoke("hello"))

full_template = """
{introduction}

{example}

{start}
"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)
example_template = """Here's an example of an interaction:
Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)
start_template = """Now, do this for real!
Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
prompt_str = pipeline_prompt.format(
    person="Elon Musk",
    example_q="What's your favorite car?",
    example_a="Tesla",
    input="What's your favorite social media site? Why?",
)
print(prompt_str)

response = llm.invoke(prompt_str)
print(response.content)



