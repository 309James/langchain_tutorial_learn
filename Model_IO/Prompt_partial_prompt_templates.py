#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 18:12
# @Author  : james.chen
# @File    : Prompt_partial_prompt_templates.py

from langchain_core.prompts import PromptTemplate

test1 = "{foo}{bar}"
prompt_1 = PromptTemplate.from_template(test1)
partial1 = prompt_1.partial(bar="ball")
print(partial1.format(foo="aaa"))