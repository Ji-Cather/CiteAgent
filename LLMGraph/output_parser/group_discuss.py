from __future__ import annotations

import re
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

from . import OutputParseError, cora_output_parser_registry



    
@cora_output_parser_registry.register("group_discuss")
class GroupDiscussParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        
        try:
            last_period_index = llm_output.rfind('.')
            if last_period_index != -1:
                llm_output = llm_output[:last_period_index + 1]
               
            return AgentFinish(return_values={"return_values":{"communication":llm_output}},
                               log=llm_output)
        except Exception as e:
            raise OutputParseError(f"Output Format Error (group discuss)")
    

    
@cora_output_parser_registry.register("choose_researcher")
class ChooseResearcherParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            output = llm_output.strip().split("\n")[0]
            return AgentFinish(return_values={"return_values":{"researcher":output}},
                               log=output)
        except Exception as e:
             raise OutputParseError("Output Format Error (choose researcher)")
         
@cora_output_parser_registry.register("get_topic")
class GetTopicParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        llm_output +="\n"
        regex = r"Action.*?:(.*?)\n"
        match = re.search(regex, llm_output, re.DOTALL|re.IGNORECASE)
        try:
            action = match.group(1).strip().lower()
            if action == "writetopic":
                
                regex = r"Thought\s*\d*\s*:(.*?)\nAction.*?:(.*?)\nTopic.*?:(.*?)\nKeywords.*?:(.*?)\nAbstract.*?:(.*?)\nFinish.*?:(.*?)\n"
                output = re.search(regex,llm_output,re.DOTALL|re.IGNORECASE)
                finish = output.group(6)
                if "true" in finish.lower():
                    finish = True
                else:
                    finish = False
                return AgentFinish(return_values={"return_values":{"thought":output.group(1),
                                                                   "action":"writetopic",
                                                                   "topic":output.group(3),
                                                                   "keyword":output.group(4),
                                                                   "finish":finish,
                                                                   "abstract":output.group(5)}},
                                log = llm_output)
            elif action == "discuss":
                return AgentFinish(return_values={"return_values":{"action":"discuss"}},
                                log=llm_output)
                
        except Exception as e:
             raise OutputParseError("Output Format Error (topic generation)")
        
   