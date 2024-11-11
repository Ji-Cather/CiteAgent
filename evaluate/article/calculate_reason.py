import os
import shutil
import json
import yaml
import time
import openai
from LLMGraph.utils.io import readinfo,writeinfo





def calculate_reason(article_meta_data,save_path ):
    # if os.path.exists(save_path):
    #     return readinfo(save_path)
    cited_reasons = {} # title: reason, section: []
    for article_title,article_data in article_meta_data.items():
        if 'motive_reason' not in article_data or 'part_reason' not in article_data:
            continue
        try:
            citation_reasons = article_data['motive_reason']
            part_reasons = article_data['part_reason']
            part_placements = article_data['part_placements']
        except:
            continue

        for cite_info in part_reasons:
            try:                
                if article_title not in cited_reasons.keys():
                    cited_reasons[article_title] = {
                        "motive_reason":{},
                        "part_reason":{},
                        "section":{},
                        "importance":[],
                    }

                for reason_id in cite_info["reason"]:
                    if reason_id not in cited_reasons[article_title]["part_reason"]:
                        try:
                            int(reason_id)
                        except:
                            continue
                        cited_reasons[article_title]["part_reason"][reason_id] = 0
                    cited_reasons[article_title]["part_reason"][reason_id] +=1
            except Exception as e:
                continue

        for cite_info in citation_reasons:
            try:                
                if article_title not in cited_reasons.keys():
                    cited_reasons[article_title] = {
                        "motive_reason":{},
                        "part_reason":{},
                        "section":{},
                        "importance":[],
                    }

                for reason_id in cite_info["reason"]:
                    if reason_id not in cited_reasons[article_title]["motive_reason"]:
                        try:
                            int(reason_id)
                        except:
                            continue
                        cited_reasons[article_title]["motive_reason"][reason_id] = 0
                    cited_reasons[article_title]["motive_reason"][reason_id] +=1
            except Exception as e:
                continue

        for cite_info in part_placements:
            try:                
                if article_title not in cited_reasons.keys():
                    cited_reasons[article_title] = {
                        "motive_reason":{},
                        "part_reason":{},
                        "section":{},
                        "importance":[],
                    }
                for reason_id in cite_info["reason"]:
                    if reason_id not in cited_reasons[article_title]["section"]:
                        try:
                            int(reason_id)
                        except:
                            continue
                        cited_reasons[article_title]["section"][reason_id] = 0
                    cited_reasons[article_title]["section"][reason_id] +=1

                count_col_names = ["importance"]
                for count_col_name in count_col_names:
                    cited_reasons[article_title][count_col_name].append(
                            cite_info[count_col_name]
                        )
            except Exception as e:
                continue

        
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    writeinfo(save_path,cited_reasons)
    return cited_reasons


# fig 4: reason 核心国家的论文 被引用是 哪类part reason #按照国家分cite reason?
# fig 4: 截至某个时间点 某个论文选择引用其他国家的part reason分布