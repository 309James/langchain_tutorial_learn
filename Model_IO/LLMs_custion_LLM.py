#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 18:23
# @Author  : james.chen
# @File    : LLMs_custion_LLM.py
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class CustomLLM(LLM):
    n:int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permmited")
        return prompt[:self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"n":self.n}


if __name__ == '__main__':
    llm = CustomLLM(n=20)
    print(llm)