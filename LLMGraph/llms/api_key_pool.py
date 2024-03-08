
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from pydantic import BaseModel

import copy


def load_llm(**llm_config):
    llm_config_temp = copy.deepcopy(llm_config)
    llm_type = llm_config_temp.pop('llm_type', 'text-davinci-003')
    if llm_type == 'gpt-3.5-turbo':
        return ChatOpenAI(model_name= "gpt-3.5-turbo",
                          **llm_config_temp)
    elif llm_type == 'text-davinci-003':
        return OpenAI(model_name="text-davinci-003",
                      **llm_config_temp)
    elif llm_type == 'gpt-3.5-turbo-16k-0613':
        return ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613",
                      **llm_config_temp)  
    elif llm_type =="gpt-3.5-turbo":
         return ChatOpenAI(model_name="gpt-3.5-turbo",
                      **llm_config_temp)  
    elif llm_type =="gpt-4":
         return ChatOpenAI(model_name="gpt-4",
                      **llm_config_temp)  
    else:
        #return OpenAI(**llm_config)
        raise NotImplementedError("LLM type {} not implemented".format(llm_type))




class APIKeyPool(BaseModel):
    
    available_keys: set = ()
    in_use_keys: set = ()    
    llm_data_path:str = "LLMGraph/llms/api.json"
    
    
    def __init__(self,
                 llm_data_path = "LLMGraph/llms/api.json"):
        
        with open(llm_data_path,'r',encoding = 'utf-8') as f:
            keys=json.load(f)

        super().__init__(
            available_keys = set(keys),
            in_use_keys = ()    ,
            llm_data_path = llm_data_path
        )
        
    def save_apis(self):
        apis = [*self.available_keys,*self.in_use_keys]
        with open(self.llm_data_path,'w',encoding = 'utf-8') as f:
            json.dump(apis, f, indent=4,separators=(',', ':'),ensure_ascii=False)
       
    
    def get_llm(self,
                llm_configs):
        """_summary_

        Args:
            tenant (_type_): _description_

        Returns:
            _type_: _description_
        """
       
            
        if len(self.available_keys) == 0:
            self.available_keys = self.in_use_keys
            self.in_use_keys = set()
            
        if len(self.available_keys) == 0:
            raise Exception("No valid OPENAI_API_KEY !!!")
        
        key = self.available_keys.pop()
        self.in_use_keys.add(key)
        
        return self.llm(key,**llm_configs)
    
    
    
    def get_llm_single(self,llm_configs):

        if len(self.available_keys) == 0:
            self.available_keys = self.in_use_keys
            self.in_use_keys = set()
            
        if len(self.available_keys) == 0:
            raise Exception("No valid OPENAI_API_KEY !!!")
            
        key = self.available_keys.pop()
        self.in_use_keys.add(key)
        return self.llm(key,**llm_configs)

    def release_llm(self, tenant=None):
       
        if len(self.in_use_keys)>0:
            key = self.in_use_keys.pop()
            self.available_keys.add(key)
            
            
    def invalid(self,
                api_key):
        print(f"{api_key} expires!!")
        if api_key in self.available_keys:
            self.available_keys.remove(api_key)
        if api_key in self.in_use_keys:
            self.in_use_keys.remove(api_key)
        print(f"{api_key} removed from pool!!")
            
    def llm(self,
            api,
            **llm_config,
            ):
       
        llm_config["openai_api_key"] = api
        return load_llm(**llm_config)

    
