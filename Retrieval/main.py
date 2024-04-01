#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 22:07
# @Author  : james.chen
# @File    : main.py
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from tools import WebSearch, WebExtractAndSummarize



if __name__ == '__main__':
    llm = ChatOpenAI()
    websites = WebSearch('')
    # websites = ["https://www.wsj.com"]
    summary = WebExtractAndSummarize(websites)
    pass