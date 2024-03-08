import json
from json import JSONDecodeError
from typing import List, Union,Sequence,Literal

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import AgentOutputParser
from . import OutputParseError, output_parser_registry

import re


from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser,OpenAIToolAgentAction
    
class AgentActionStep(Serializable):
    return_values: dict
    """Dictionary of return values."""
    log: str
    type: Literal["AgentActionStep"] = "AgentActionStep"  # type: ignore
    
    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Convert an agent action to a message.

        This code is used to reconstruct the original AI message from the agent action.

        Args:
            agent_action: Agent action to convert.

        Returns:
            AIMessage that corresponds to the original tool invocation.
        """
        return [HumanMessage(content=self.log)]



@output_parser_registry.register("write_article")
class WriteArticleParser(OpenAIToolsAgentOutputParser):


    @property
    def _type(self) -> str:
        return "write_article_parser"
    
    @staticmethod
    def parse_ai_message_to_openai_tool_action(
        message: BaseMessage,
    ) -> Union[List[AgentAction], AgentFinish]:
        """Parse an AI message potentially containing tool_calls."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        if not message.additional_kwargs.get("tool_calls"):
            # return AgentFinish(
            #     return_values={"output": message.content}, log=str(message.content)
            # )
            llm_output = message.content
            llm_output += "\n"
            regex = r"Title\s*\d*\s*:(.*?)\nKeywords.*?:(.*?)\nAbstract.*?:(.*?)\nCitations.*?:(.*)"
            
            try:
                output = re.search(regex, llm_output, re.DOTALL|re.IGNORECASE)
                # finish = output.group(5)
                # if "true" in finish.lower():
                #     finish = True
                # else:
                #     finish = False
                article_infos ={"title":output.group(1).strip(),
                                "keyword":output.group(2).strip(),
                                "abstract":output.group(3).strip(),
                                "citation":output.group(4).strip()}
               
                return AgentFinish(return_values={"return_values":article_infos},
                                log = llm_output)
            except Exception as e:
                print(e)
                return AgentOutputParser("error for write article")

        actions: List = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            function = tool_call["function"]
            function_name = function["name"]
            try:
                _tool_input = json.loads(function["arguments"] or "{}")
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function} because "
                    f"the `arguments` is not valid JSON."
                )

            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            actions.append(
                OpenAIToolAgentAction(
                    tool=function_name,
                    tool_input=tool_input,
                    log=log,
                    message_log=[message],
                    tool_call_id=tool_call["id"],
                )
            )
        return actions
            
    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self.parse_ai_message_to_openai_tool_action(message=message)
    