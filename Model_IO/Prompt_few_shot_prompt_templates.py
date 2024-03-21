#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 15:41
# @Author  : james.chen
# @File    : Prompt_few_shot_prompt_templates.py
"""
In this tutorial, we’ll learn how to create a prompt template that uses few-shot examples.
A few-shot prompt template can be constructed from either a set of examples, or from an Example Selector object.
In this tutorial, we’ll configure few-shot examples for self-ask with search.
"""
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# 很多自问自答的例子
examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How old was Muhammad Ali when he died?
    Intermediate answer: Muhammad Ali was 74 years old when he died.
    Follow up: How old was Alan Turing when he died?
    Intermediate answer: Alan Turing was 41 years old when he died.
    So the final answer is: Muhammad Ali
    """,
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who was the founder of craigslist?
    Intermediate answer: Craigslist was founded by Craig Newmark.
    Follow up: When was Craig Newmark born?
    Intermediate answer: Craig Newmark was born on December 6, 1952.
    So the final answer is: December 6, 1952
    """,
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who was the mother of George Washington?
    Intermediate answer: The mother of George Washington was Mary Ball Washington.
    Follow up: Who was the father of Mary Ball Washington?
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
    So the final answer is: Joseph Ball
    """,
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who is the director of Jaws?
    Intermediate Answer: The director of Jaws is Steven Spielberg.
    Follow up: Where is Steven Spielberg from?
    Intermediate Answer: The United States.
    Follow up: Who is the director of Casino Royale?
    Intermediate Answer: The director of Casino Royale is Martin Campbell.
    Follow up: Where is Martin Campbell from?
    Intermediate Answer: New Zealand.
    So the final answer is: No
    """,
    },
]

# Create a formatter for the few-shot examples
example_prompt = PromptTemplate(input_variables=["question", "answer"],
                                template="Question: {question}\nAnswer: {answer}")
# print(example_prompt.format(**examples[0]))

# feed examples directly to FewShotPromptTemplate

# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     suffix="Question: {input}",
#     input_variables=["input"]
# )
# print(prompt.format(input="Who is the best NBA player? Lebron James or Micheal Jordan?"))

# using the example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

question = "Who is the best NBA player? Lebron James or Micheal Jordan?"
selected_examples = example_selector.select_examples({"question:": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print('\n')
    for k, v in example.items():
        print(f"{k}: {v}")
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)
print(prompt.format(input="Who is the best NBA player? Lebron James or Micheal Jordan?"))
