# Modified from langchain.memory.summary.py

from typing import Any, Dict, List, Tuple, Type, Union

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import (AgentAction, AIMessage, BaseMessage, ChatMessage,
                              SystemMessage, get_buffer_string)
from pydantic import BaseModel, root_validator
from LLMGraph.message import Message

from openai import RateLimitError,AuthenticationError
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from . import memory_registry

class SummarizerMixin(BaseModel):
    llm: BaseLanguageModel
    
    llm_loader:Any
    memory_llm_configs:dict = {}
    max_retry :int = 5
    
    def reset_llm(self, llm):
        self.llm = llm
    
    def request(self,
                chain:LLMChain,
                **kargs):
        response = None
        for i in range(self.max_retry):
            try:
                response = chain.predict(**kargs)
                if response is not None:
                    break
            except AuthenticationError as e:
                    if isinstance(self.llm,OpenAI) or isinstance(self.llm,ChatOpenAI):
                        api_key = self.llm.openai_api_key
                        self.llm_loader.invalid(api_key)
                        llm = self.llm_loader.get_llm_single(self.memory_llm_configs)
                        self.llm = llm
                    print(e,"Retrying...")
                    continue
            except Exception as e:
                continue
            
        if response is None:
            return ""
        return response
    
    async def arequest(self,
                       chain:LLMChain,
                       **kargs):
        
        response = None
        
        for i in range(self.max_retry):
            try:
                response = await chain.apredict(**kargs)
                if response is not None:
                    break
            except AuthenticationError as e:
                    if isinstance(self.llm,OpenAI) or isinstance(self.llm,ChatOpenAI):
                        api_key = self.llm.openai_api_key
                        self.llm_loader.invalid(api_key)
                        llm = self.llm_loader.get_llm_single(self.memory_llm_configs)
                        self.llm = llm
                    print(e,"Retrying...")
                    continue
            except Exception as e:
                continue
            
        if response is None:
            return ""
        return response

    def predict_new_summary(
        self, 
        messages: List[Message], 
        existing_summary: str,
        prompt: BasePromptTemplate = SUMMARY_PROMPT
    ) -> str:
        new_lines = "\n".join([str(message) for message in messages])

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return self.request(chain,summary=existing_summary, new_lines=new_lines)
    
    
    async def apredict_new_summary(
        self, 
        messages: List[Message], 
        existing_summary: str,
        prompt: BasePromptTemplate = SUMMARY_PROMPT
    ) -> str:
        new_lines = "\n".join([str(message) for message in messages])

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return await self.arequest(chain,summary=existing_summary, new_lines=new_lines)
    
    async def summarize_paragraph(self,
                            passage:str,
                            old_summary:str="",
                            prompt: BasePromptTemplate = SUMMARY_PROMPT):
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return await self.arequest(chain,summary="", new_lines=passage)

    async def summarize_chatting(self,
                           prompt_inputs,
                           prompt):
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return await self.arequest(chain,**prompt_inputs)

@memory_registry.register("summery_history")
class SummaryMemory(BaseChatMemory, SummarizerMixin):
    """Conversation summarizer to memory."""

    buffer: str = ""
    memory_key: str = "history"  #: :meta private:
    name :str =""
    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]


    

    def save_context(self, contexts: Union[List[Tuple[AgentAction, str]], List[Message]]) -> None:
        """Save context from this conversation to buffer."""
        for context in contexts:
            if isinstance(context, Message):
                self.chat_memory.messages.append(ChatMessage(content=context.content, role=context.sender))
            elif isinstance(context, tuple) and len(context) == 2 and \
                isinstance(context[0], AgentAction) and isinstance(context[1], str):
                self.chat_memory.messages.append(ChatMessage(content=context[0].log.strip() + '\nObservation:' + context[1], role=""))
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-len(contexts):], self.buffer
        )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""