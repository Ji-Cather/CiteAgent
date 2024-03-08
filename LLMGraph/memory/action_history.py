from typing import List,Union,Dict

from pydantic import Field

from LLMGraph.message import Message

from . import memory_registry,summary_prompt_default
from .base import BaseMemory

from langchain.chains.llm import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel, root_validator
import re
from typing import Any

from openai import RateLimitError,AuthenticationError

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
class SummarizerMixin(BaseModel):
    llm: BaseLanguageModel
    
    llm_loader:Any
    memory_llm_configs:dict = {}
    max_retry :int = 5
    
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


@memory_registry.register("action_history")
class ActionHistoryMemory(BaseMemory,SummarizerMixin):
    messages: Dict[str,List[Message]] = {} # 包括自己做的行为+自己想发的信息
    summarys: Dict[str,Message] = {} # 记录某类记忆的summary
    buffer_step: Dict[str,int] = {} # 记录某类mem，总结到哪个位置（index)
    summary_prompt:dict = {"community":"community_summary",
                      "house_type":"house_type_summary",
                      "house":"house_summary",
                      "search":"forum_search_summary",
                      "publish":"forum_publish_summary"}
    
    # 后者代表最新的所有信息合成的summary
    multi_message_summary:str = "" # 设置为buffer，每次做不同信息合成都会用到
    new_message:bool = False # 是否存在新的message加入记忆
    reflection:bool = False # 若设置为true,则触发分类reflection  
    
    # 想要发出的message信息，在发出后清空，加入messages中
    post_message_buffer: List[Message] = []
    mail:List[Message] = [] # 用于socialnetwork的接收信息
    
    social_network: dict = {} # id:{name:,view:,dialogues：List[Message]}
    
    forum: dict = {"buffer":[],
                   "chat_history":""} # untrusted, trusted, buffer
    
    
    # 设置各类消息的buffer大小数
    summary_threshold:int = 5 # 每次总结后，再多5条就触发一次总结 -> 记忆库内
    dialogue_threshold:int = 20 # 保证和各个熟人的记忆都在20条内（按照时间戳） -> social_network内
    

    
    def reset_llm(self, llm):
        self.llm = llm
    
    def __init__(self,**kwargs):
        social_network = kwargs.pop("social_network")
        for id,info in social_network.items():
            info["dialogues_pt"] = -1       
            info["chat_history"] = ""
            info["comment"] = ""
            info["processed"] = []

        
        super().__init__(social_network = social_network,
                         **kwargs)
    
    def receive_message(self, 
                    messages: List[Message]):
        self.mail.extend(messages)
        self.add_social_network_message(messages,receive = True)
    
    def post_meesages(self):
        # for message in self.post_message_buffer:
        #     self.add_message([message])
        post_messages = self.post_message_buffer.copy()
        self.post_message_buffer = []
        return post_messages
        
    def add_post_meesage_buffer(self, 
                    messages: List[Message]):
        self.post_message_buffer.extend(messages)
 
    
    def add_message(self, 
                    messages: List[Message],
                    receive = False) -> None:   

        messages_pt = self.messages
        buffer_step_pt = self.buffer_step
            
        for message in messages:
            if message.message_type in messages_pt.keys():
                if (message) not in messages_pt[message.message_type]: # 防止重复append social_network前后信息
                    messages_pt[message.message_type].append(message)
                
            else:
                messages_pt[message.message_type]=[message]
                buffer_step_pt[message.message_type]=-1
                

        for type_m in messages_pt.keys():
            if len(messages_pt[type_m]) - \
                buffer_step_pt[type_m] > self.summary_threshold:
                    self.new_message = True
                    self.summary_type_memory(type_message = type_m,
                                            receive = receive)
        
    def add_forum_message(self,messages: List[Message]):
        self.forum["buffer"].append(messages)
                    
    def add_social_network_message(self, messages: List[Message],receive = False):
        if receive:
            for message in messages:
                sender_id = list(message.sender.keys())[0]
                if sender_id not in self.social_network.keys():
                    self.social_network[sender_id] = {"name":list(message.sender.values())[0],
                                                      "relation":"stranger"}
                    
                if "dialogues" in self.social_network[sender_id].keys():
                    self.social_network[sender_id]["dialogues"].append(message)
                else:
                    self.social_network[sender_id]["dialogues"] = [message]
                    self.social_network[sender_id]["dialogues_pt"] = -1
                        
                
        else: # 发出的social_net 信息
            for message in messages:
                for receiver_id in message.receivers:
                    if "dialogues" in self.social_network[receiver_id].keys():
                        self.social_network[receiver_id]["dialogues"].append(message)
                    else:
                        self.social_network[receiver_id]["dialogues"] = [message]
                        self.social_network[receiver_id]["dialogues_pt"] = -1
                        
        # check dialogue threshold
        for t_id,t_info in self.social_network.items():
            dialogues = t_info.get("dialogues",[])
            if (len(dialogues)>self.dialogue_threshold):
                dialogues.sort(key=lambda x: x.timestamp,reverse=True)
                t_info["dialogues"] = dialogues[:20]
                
        
                        

        
    def topk_message_default(self,
                             messages:List[Message],
                             k=5)->List[Message]:
        messages.sort(key=lambda x: x.sort_rate(),reverse=True)
        return messages[:k] if k<len(messages) else messages
    
    # choose类别的记忆选择："community","house_type","house"
    # 例：community的memory选择
        # 种类选择search_forum，publish_forum，choose_community 类记忆
        # 先按照时间排序所有没被总结summary的记忆，选其中前三条
        # 再添加search_forum，publish_forum，choose_community的summary记忆
    
        
    
    
    

    async def summary_synthesize_memory(self,
                                  messages:List[Message],
                                  existing_memory = "") -> str:
        prompt_template = summary_prompt_default.get("synthesize_summary","")
        prompt = PromptTemplate(input_variables=["summary", "new_lines"], 
                                template=prompt_template)
        summerize_mem = await self.apredict_new_summary(messages=messages,
                                                existing_summary=existing_memory,
                                                prompt=prompt)
        summerize_mem = summerize_mem.strip().strip(" ").strip('"')
        return summerize_mem
        
    def summary_type_memory(self,
                            type_message:Union[str,List[str]] = "all",
                            receive = False)->str:

        messages_pt = self.messages
        message_summary_pt = self.summarys
        buffer_step_pt = self.buffer_step
        
        type_messages = [type_message] if not isinstance(type_message,list) else type_message
        for type_m in type_messages:
            start_idx = buffer_step_pt[type_m] + 1
            new_messages = messages_pt[type_m][start_idx:]
            
            if receive:
                # receive的memory暂时都用这个进行summary
                prompt_template = summary_prompt_default.get("synthesize_summary","") 
            else:
                yaml_key = self.summary_prompt.get(type_m)
                prompt_template = summary_prompt_default.get(yaml_key,"")
                
            prompt = PromptTemplate(input_variables=["summary", "new_lines"], 
                                    template=prompt_template)
            summerize_mem = self.predict_new_summary(messages=new_messages,
                                                     existing_summary=message_summary_pt.get(type_m,""),
                                                     prompt=prompt)
            message_summary_pt[type_m] = Message(message_type = type_m,
                                            content = summerize_mem.strip())
            
             # 总结了所有最新信息，更新总结位置
            buffer_step_pt[type_m] = len(messages_pt[type_m]) -1
            



    def to_string(self, 
                  messages:List[Message],
                  add_sender_prefix: bool = False,
                  ) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    f"{list(message.sender.values())[0]}:{str(message)}"
                    if list(message.sender.values())[0] != ""
                    else str(message)
                    for message in messages
                ]
            )
        else:
            return "\n".join([str(message) for message in messages])
        
    def to_string_default(self, 
                  add_sender_prefix: bool = False,
                  type_message:Union[str,List[str]] = "all",
                  ) -> str:
        
        messages_return=[]
        if (type_message=="all") or ("all" in type_message):
            type_messages = ["search","publish","community","house","house_type"]

        type_messages = [type_message] if not isinstance(type_message,list) else type_message
        
        for type_m in type_messages:
            if (type_m in self.messages.keys()):
                messages_return.extend(self.messages.get(type_m,[]))
        

        messages_return = self.topk_message_default(messages_return)
        
        if add_sender_prefix:
            return "".join(
                [
                    f"[{message.sender}]: {str(message)}"
                    if message.sender != ""
                    else str(message)
                    for message in messages_return
                ]
            )
        else:
            return "".join([str(message) for message in messages_return])

    def reset(self) -> None:
        self.messages = {}
        self.summarys = {}
        self.post_message_buffer = []


    ###############         一系列的 retrive memory rule       ##############
    
    #  调用各类 retrive 方法
    async def memory_tenant(self,mem_type:str,name=None)->str:
        house_attrs =("house_type",
                      "house_orientation",
                      "floor_type")
        
        TYPE_MEM={
            "community":{"type_messages":["base","community"],"social_network":True},
            "house":{"type_messages":["base","house"],"social_network":False},
            "house_type":{"type_messages":["base","house_type"],"social_network":False},
            "house_orientation":{"type_messages":["base","house_orientation"],"social_network":False},
            "floor_type":{"type_messages":["base","floor_type"],"social_network":False},
        } # 注 forum暂时只有小区消息
        
        
        if mem_type in TYPE_MEM.keys():# 默认retrive方法
            type_messages_config = TYPE_MEM.get(mem_type,[]) 
            return await self.retrive_basic(**type_messages_config,
                                      name=name,
                                    ) # social_network 控制传不传入交流信息
            
        elif "social_network" in mem_type:# social_network retrive方法
            return await self.retrive_all_memory(name)
        elif "publish" == mem_type or "search" == mem_type: 
            return await self.retrive_all_memory(name)
        # elif mem_type == "social_network":# social_network retrive方法
        #     return self.retrive_group_discuss_plan_memory(name)
        #     # return self.retrive_group_discuss_memory()
        # elif mem_type =="social_network_message_back":
        #     return self.retrive_group_discuss_plan_memory(name)
            # return self.retrive_group_discuss_message_back_memory()
        elif mem_type == "relation":
            return await self.retrive_relation_memory()
        else: 
            return ""
        
    def retrieve_recent_chat(self,
                             tenant_ids: Union[List,str] = "all"):
        if tenant_ids == "all":
            tenant_ids = list(self.social_network.keys())
            
        recent_chats = []
        
        for tenant_id in tenant_ids:
            if ("dialogues" in self.social_network[tenant_id].keys()):
                dialogue_sn = self.social_network[tenant_id].get("dialogues")
                if isinstance(dialogue_sn,Message):
                    context_sn = dialogue_sn.context
                    recent_chats.append(context_sn)
                
        return "\n".join(recent_chats)
        
    
    # 默认retrive方法
    async def retrive_basic(self,
                      type_messages:List[str],
                      name:str = None,
                      social_network = False):
        memory_return =""
        if "base" in type_messages:
            memory_return += self.to_string_default(add_sender_prefix=False,
                                        type_message=["base"])
            type_messages.remove("base")
        
        if not self.reflection:
            memory_return += self.to_string_default(add_sender_prefix=False,
                                        type_message=type_messages)
        else:
            messages_left = []
            for type_m in type_messages:
                if (type_m in self.messages.keys()):
                    messages_left.extend(self.messages[type_m][self.buffer_step[type_m]+1:])
            
            messages_left.sort(key=lambda x: x.timestamp,reverse=True)
            # 零散记忆限制数量为10
            messages_left = messages_left[:5] if len(messages_left) >5 else messages_left
            
            # 零散记忆需要summary吗?暂时没有归到summary的记忆中
            if len(messages_left)>0:
                # messages_str_left = await self.summary_synthesize_memory(messages_left)
                messages_str_left = self.to_string(messages_left,add_sender_prefix=False)
            else:messages_str_left = ""
            
            messages_summary = []
            for type_m in type_messages:
                if (type_m in self.summarys.keys()):
                    messages_summary.append(self.summarys.get(type_m,))
            
            memory_str_summary = self.to_string(messages=messages_summary,
                                  add_sender_prefix=False)
            
            memory_return += (messages_str_left + memory_str_summary)
        
        
        if social_network:
            
            # 在social_net 不加入信息时，不进行更改（房子的记忆改了也不进行更改）
            type_messages = ["community","house","house_type","publish"]
            memory_house_info = await self.reflect_memory_multi_types(type_messages) 
            memory_discussion = await self.reflect_memory_forum(memory_house_info,name)
            memory_discussion += await self.reflect_memory_social_network(memory_house_info,
                                                                    name) 

            memory_return += memory_discussion # 这里 memory_house_info 不会加入记忆，
            
        return memory_return # 结构为 type_messages(零散记忆+总结记忆) + social_network
        
        
    
    
    
    async def reflect_memory_multi_types(self,type_messages:list) -> str :
        
        if not self.reflection:
            return self.to_string_default(add_sender_prefix=False,
                                        type_message=type_messages)
            
        if not self.new_message: # 这边的new message 可以明确到哪些是增加的message
           
            if self.multi_message_summary == "": # 如果二级summary为空，则会初始化一个
                # 当一级summary都空的时候，不进行summary
                messages_str = []
                for type_m in type_messages:
                    if (type_m in self.summarys.keys()):
                        messages_str.append(self.summarys.get(type_m))
                if len(messages_str) > 0:
                    self.multi_message_summary = await self.summary_synthesize_memory(messages_str,existing_memory=self.multi_message_summary)
                    
            return self.multi_message_summary
        
        # memory_house_info 部分
        messages_str = []
        for type_m in type_messages:
            if (type_m in self.messages.keys()):
                messages_str.extend(self.messages[type_m][self.buffer_step[type_m]+1:])
                
        messages_str.sort(key=lambda x: x.timestamp,reverse=True)
        messages_str = messages_str[:5] if len(messages_str)>5 else messages_str
        
        for type_m in type_messages:
            if (type_m in self.summarys.keys()):
                messages_str.append(self.summarys.get(type_m))
                
        
        self.multi_message_summary = await self.summary_synthesize_memory(messages_str,existing_memory=self.multi_message_summary)
        self.new_message = False
        return self.multi_message_summary # 二级summary
    
    
    async def reflect_memory_forum(self,
                             memory_house_info:str,
                             name :str = None):
        
        if len(self.forum["buffer"]) < self.summary_threshold:
            if  self.forum["chat_history"] != "":
                content_return  = self.forum["chat_history"]
                new_forum_info = self.to_string_default(self.forum["buffer"])    
                content_return += "Here's the latest information you get from forum:{forum}".format(forum = new_forum_info)
                return content_return
            else:
                return "Here's the latest information you get from forum:{forum}".format(\
                    forum = self.to_string_default(self.forum["buffer"]))
            
        
        
        # 这里assess 从forum中搜索search到的记忆 
        prompt_template = summary_prompt_default.get("forum_assess_summary")
        prompt = PromptTemplate(input_variables=["name", 
                                                 "memory",
                                                 "forum_info"], 
                                    template=prompt_template)
        
        
        new_forum_info = self.to_string_default(self.forum["buffer"])
        current_summary = self.forum["chat_history"]
        
        memory_forum = "Here' your memory:{memory}\nHere's your previous summary of forum information: {info}".format(memory = memory_house_info,info= current_summary)                
        
        prompt_inputs = {
                "name": name,
                "memory": memory_forum,
                "forum_info": new_forum_info,
            }
            
        
        forum_assess_response = await self.summarize_chatting(prompt_inputs=prompt_inputs,
                                                            prompt=prompt)
            
        try:
            # Parse out the action and action input
            forum_assess_response += "\n"
            regex = r".*?Trusted\.*?:(.*?)\n.*?Suspicious\.*?:(.*?)\n.*?Reason.*?:(.*?)\n"
            match = re.search(regex, forum_assess_response, re.DOTALL)
            
            infos = {"trusted_info":match[1],"reason_guess":match[3]}
            if "none" not in match[2].lower():
                infos["untrusted_info"] =  match[2]
                
            if "processed" not in self.forum.keys():
                self.forum["processed"] = []
            self.forum["processed"].append(infos)
                

            if "untrusted_info" in infos.keys():
                content_template = """You trust these information on forum: {trusted_info} \
However you are suspicious of {untrusted_info} Because {reason_guess}"""
            else:
                content_template = """You trust these information on forum: {trusted_info} Because {reason_guess}"""
            
            self.forum["chat_history"] = content_template.format_map(infos)
            return self.forum["chat_history"]
                
        except Exception as e:
            print("fail to parse social_network memory summary") 
            return forum_assess_response

            

    
    async def reflect_memory_social_network(self,
                                      memory_house_info :str,
                                      name :str = None):
        
        def format_infos(infos:dict,
                         acquaintance_info:dict):
            content_templates = {
                        "untrusted_info":" However you are suspicious of {untrusted_info}",
                        "trusted_info":"You trust {ac_name} said that {trusted_info}",
                        "reason_guess":"Because {reason_guess}",
                        "fail_process":"{fail_process}"
                    }    
            content_template = "Your summary of chatting history with {ac_name}:"
            for key in infos.keys():
                content_template += content_templates[key]
            
            infos.update({"ac_name": acquaintance_info["name"]})              
            memory_discussion_acquaintance = content_template.format_map(infos)
            return memory_discussion_acquaintance
        
        NOTE ="""Keep this in mind: you and your acquaintances are in the same renting system. \
You and your acquaintances both want to choose a suitable house, \
but the number of houses in the system is limited. You are in a competitive relationship with each other."""

        memory_discussion = ""
        prompt_template = summary_prompt_default.get("social_network_summary")
        prompt = PromptTemplate(input_variables=["name", 
                                                 "acquaintance_description",
                                                 "memory",
                                                 "acquaintance_name",
                                                 "dialogue"], 
                                    template=prompt_template)
        
        
        for acquaintance_id in list(self.social_network.keys()):
            acquaintance_info = self.social_network[acquaintance_id]
           
            if "dialogues" not in acquaintance_info.keys() :
                continue # 没对话
            
            if len(acquaintance_info.get("dialogues"))- 1 == acquaintance_info["dialogues_pt"]: 
                memory_discussion += acquaintance_info.get("chat_history","")
                continue
            
            else: 
                
                infos = {}
                # 从process好的消息库中取历史内容
                for processed_info in self.social_network[acquaintance_id].get("processed",[]):
                    for k, v in processed_info.items():
                        if k not in infos.keys():
                            infos[k] =[v]
                        else:
                            infos[k].append(v)
                            
                new_dialogues = acquaintance_info.get("dialogues",[])
                new_dialogues = new_dialogues[acquaintance_info["dialogues_pt"]+1:]
                
                new_dialogues.sort(key=lambda x: x.timestamp,reverse=True)
                new_dialogues = new_dialogues[:10] if len(new_dialogues)>10 else new_dialogues
                
                latest_send_message = None
                chats = []
                for message in new_dialogues: # 这里fifo,取最近十个message
                    assert isinstance(message,Message)
                    if name in message.sender.values():
                        latest_send_message = message
                    if message.conver_num == 0 and list(message.sender.values())[0] == name: # 自己发的第一句话不管
                        continue
                    else:
                        chats.append("\n".join(message.context))

                
                
                if (len(chats)>0):
                    chats_str = "\n".join(chats)
                    acquaintance_description = """Your relationship with {ac_name} is {ac_type}. {comment} """
                
                    comment = self.social_network[acquaintance_id].get("comment","")
                    if comment != "":
                        comment = "and your comment on him is: {}".format(comment)
                
                    acquaintance_description = acquaintance_description.format(ac_name=acquaintance_info["name"],
                                                                            ac_type=acquaintance_info.get("relation"),
                                                                            comment=comment)
                    
                    memory_acquaintance = memory_house_info
                    if acquaintance_info.get("chat_history","") !="":
                        
                        memory_acquaintance += "Here's your previous summary of chatting dialogue with {ac_name}: {info}".format(
                        ac_name = acquaintance_info["name"],
                        info = acquaintance_info.get("chat_history",""))
                    
                    if latest_send_message is not None: # 如果本人是message的发送者时（消息不是收到的）
                        memory_latest_word = """\nHere's your latest response to {acquaintance_name}\n{response}"""
                        assert "plan" in latest_send_message.content.keys()
                        if "thought" in latest_send_message.content.keys():
                            response_template ="""Your previous plan to communicate: {plan}.\
    You thought "{thought}". And You said "{output}". """
                        else:
                            response_template ="""Your previous plan to communicate: {plan}. And You said "{output}". """
                        
                        memory_latest_word = memory_latest_word.format(acquaintance_name=acquaintance_info["name"],
                                                                    response=response_template.format_map(latest_send_message.content))
                        memory_acquaintance += memory_latest_word
                        
                    prompt_inputs = {
                        "name":name,
                        "acquaintance_description":acquaintance_description,
                        "memory":memory_acquaintance,
                        "acquaintance_name": acquaintance_info["name"],
                        "dialogue": chats_str
                    }
                    
                    memory_discussion_acquaintance = await self.summarize_chatting(prompt_inputs=prompt_inputs,
                                                                                prompt=prompt)
                    try:
                        assert isinstance(memory_discussion_acquaintance,str)
                    except:
                        memory_discussion_acquaintance = ""
                    
                                
                    try:
                        infos_new = {}
                        # Parse out the action and action input
                        memory_discussion_acquaintance += "\n"

                        regex_trusted = r".*?Trusted\.*?:(.*?)\n"
                        match = re.search(regex_trusted, memory_discussion_acquaintance, re.DOTALL)
                        if match is not None: infos_new.update({"trusted_info":match[1].strip()})
                        
                        regex_suspicious = r".*?Suspicious\.*?:(.*?)\n"
                        match = re.search(regex_suspicious, memory_discussion_acquaintance, re.DOTALL)
                        
                        if match is not None: 
                            unt_info = match[1].strip().strip(".")
                            if unt_info.lower() != "none":
                                infos_new.update({"untrusted_info":unt_info})
                        
                        regex_reason = r".*?Reason\.*?:(.*?)\n"
                        match = re.search(regex_reason, memory_discussion_acquaintance, re.DOTALL)
                        if match is not None: infos_new.update({"reason_guess":match[1].strip()})
                    
                    except Exception as e:    
                        
                        if infos_new!={}:
                            self.social_network[acquaintance_id]["processed"].append(infos_new)
                        else:
                            self.social_network[acquaintance_id]["processed"].append({"fail_process":memory_discussion_acquaintance})
                                         
                    
                    for k,v in infos_new.items():
                        if k not in infos.keys():
                            infos[k] = [v]
                        else:
                            infos[k].append(v)
                    
                    for k, v in infos.items():
                        infos[k] = "".join(v)
                        
                    
                    memory_discussion_acquaintance = format_infos(infos=infos,
                                                                  acquaintance_info=acquaintance_info)
                   
                    
                
                    self.social_network[acquaintance_id]["chat_history"] = memory_discussion_acquaintance
                    self.social_network[acquaintance_id]["dialogues_pt"] = len(acquaintance_info.get("dialogues"))-1
                    memory_discussion += memory_discussion_acquaintance
                else:
                    memory_discussion += acquaintance_info.get("chat_history","")
                
            
            
            

        # debug
        print("DEBUG: {name}'s summary of his/her chatting dialogues:\n{dialogue}".format(name=name,
                                                                                        dialogue=memory_discussion))
        return memory_discussion+f"\n{NOTE}"
    
    


    
    
    # needs modification; 需要summary
    # 这里会把所有social network中的信息汇总到memory中
    # 比较 time_consuming, 为了避免不必要的调用，加入了buffer指针，buffer
    
    # 这里retrieve 所有messages内["search","publish","community","house","house_type"] 总结得到的信息
    #   加上social_network中各人提供的 “我”认为 真、假消息 以及 说话者的目的 
    async def retrive_all_memory(self,
                            name = None):

        type_messages = ["community","house","house_type","publish"]
        memory_house_info = await self.reflect_memory_multi_types(type_messages)        

        
        memory_discussion = await self.reflect_memory_forum(memory_house_info,name)
        
        # 在social_net 不加入信息时，不进行更改（房子的记忆改了也不进行更改）
        memory_discussion = await self.reflect_memory_social_network(memory_house_info,name) 
        
        return memory_house_info.strip("\n") + memory_discussion
    
    
    # 这里的重点是保证内容真实可信（对于memory的主体而言）
    async def retrive_relation_memory(self):
        trusted_social_network_memory = ""
        for agent_id,agent_info in self.social_network.items():
            trusted_memory = agent_info.get("trusted_info",[])
            trusted_memory = " ".join(trusted_memory)
            trusted_social_network_memory += trusted_memory
        
        type_messages = ["search","publish","community","house","house_type"]
            
        
        return trusted_social_network_memory + await self.reflect_memory_multi_types(type_messages)
     
        
