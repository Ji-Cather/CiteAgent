import os

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import torch

from bs4 import BeautifulSoup

import json
def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)


def load_cora_raw_data(data_dir,data_name):
    path = os.path.join(data_dir,data_name)
    data = torch.load(path)
    # titles = []
    # for raw_data in data.raw_texts:
    #     title = []
    #     words = raw_data.split(" ")
    #     for word in words:
    #         if str.istitle(word) or word in [":","or","in","and","of","versus","with",]:
    #             title.append(word)
    #         else:
    #             break
    #     title = " ".join(title)
    #     strip_words =[
    #         "The","This","In","we",""
    #     ]
    #     # request_title_author(title)
    #     titles.append(title)
    writeinfo("titles.json",data.raw_texts)
    

def request_title_author(title):
    import requests
    import time
    from lxpy import copy_headers_dict

    h = copy_headers_dict('''
    cookie: sid=m8hIX......f0iLA2TZs; captui=MDdkYWViMWE5Y.......MWNBa0lqUGQ%3D; 
    upgrade-insecure-requests: 1
    user-agent: Mozilla/5.0 (Win...... Safari/537.36
    ''')


    q = title

    url = f'https://www.researchgate.net/search/publication?q={q}'
    headers = {
            'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0'
        }
    session = requests.Session()
    password ="2001012720010127jjr_%?"

    params = {'login': '2023100839@ruc.edu.cn', 'password': password}
    
    
    s = ''
    while s == '':
        try:
            s = session.post("https://www.researchgate.net/application.Login.html", data = params, headers = headers)
            break
        except Exception as e:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue

    print(s.cookies.get_dict())
    print(s.text)
    s = session.get("https://www.researchgate.net/home")
    
    if s.status_code == 429:
        print("cookies validation failed")
        # 接下来应该进行验证，获取 cattui 后构建cookie，再次请求
        # capUrl = f'https://www.researchgate.net/application.ClientValidation.html?origPath=/search/researcher?q={q}'
        # cookies = requests.utils.dict_from_cookiejar(d.cookies)
        # d = requests.get(capUrl, headers=h,cookies=cookies)
        # 但是因为没外网，获取不到验证，这里采用其他方式获取新cookie，记得修改 executable_path
        from selenium import webdriver
        driver = webdriver.Chrome(executable_path=r'/data/jiarui_ji/article_agent/chromedriver')
        driver.get(url)
        time.sleep(5)
        cookies = {}
        for cook in driver.get_cookies():
            cookies[cook['name']]=cook['value']

    soup = BeautifulSoup(d.text,"html.parser")
    all_articles = soup.find_all(name='a', class_="nova-legacy-e-link nova-legacy-e-link--color-inherit nova-legacy-e-link--theme-bare")
    soup.find_all(name='a',attrs ={
        "href":True
    },recursive=True)
    for article in all_articles:
        pass


def load_cora_data():
    # Specify the path to the CORA dataset
    data_path = "/data/jiarui_ji/graph_data/cora"
    edges_path = os.path.join(data_path, "cora.cites")
    content_path = os.path.join(data_path, "cora.content")

    # Read the CORA dataset content
    content = np.genfromtxt(content_path, dtype=np.dtype(str))
    ids = content[:, 0].astype(np.int32)
    features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
    labels = content[:, -1]

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Read the edges
    edges = np.genfromtxt(edges_path, dtype=np.int32)

    return features, labels, edges

load_cora_raw_data("/home/runlin_lei/LLMGIA/data","citeseer_fixed_sbert.pt")
