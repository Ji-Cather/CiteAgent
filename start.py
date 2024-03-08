import os
import shutil
import json
import yaml
import time
import openai
def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list



def run_tasks(tasks,
              data,
              log_dir,
              run_ex_times = 1):

    
    
    success_tasks = []
    
    failed_tasks = []
    
    command_template = "python main.py --task {data} --config {task} --simulate"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_tasks_dir = os.path.join(log_dir,"log")
    if not os.path.exists(log_tasks_dir):
        os.makedirs(log_tasks_dir)

    complete_path = os.path.join(log_dir,"complete.json")            
    
    
    for idx,task in enumerate(tasks):
        
        task_root_path = os.path.join("LLMGraph/tasks",data,"configs",task,"result")

        done_times = 0
        if os.path.exists(task_root_path):
            results = os.listdir(task_root_path)

            for result in results:
                result_path = os.path.join(task_root_path,result)
                if os.path.exists(os.path.join(result_path,"all")):
                    done_times +=1
            
        iter_p = 0
        max_iter = run_ex_times
            
        while(done_times<run_ex_times and iter_p<max_iter):
            task = task.replace("(","\(").replace(")","\)")
            log_task_path = os.path.join(log_tasks_dir,f"{task}.log")
            command = command_template.format(task = task,
                                            data = data,
                                            log_path = log_task_path)
            
            try:
                return_val = os.system(command)
                if return_val ==0:
                    done_times+=1
            except Exception as e:
                print(e)
                
            iter_p+=1
            
        if (done_times == run_ex_times):
            success_tasks.append(task.replace("\(","(").replace("\)",")"))
        else:
            failed_tasks.append(task.replace("\(","(").replace("\)",")"))
   
        with open(complete_path,'w',encoding = 'utf-8') as f:
            uncomplete_tasks =tasks[idx+1:] if (idx+1)< len(tasks) else []
            json.dump({"success":success_tasks,
                    "failed":failed_tasks,
                    "uncomplete":uncomplete_tasks}, 
                    f,
                    indent=4,
                    separators=(',', ':'),ensure_ascii=False)
        
def run_tasks_logs(data = "PHA_51tenant_5community_28house",
                   configs:list = None
                   ):
    config_root = f"LLMGraph/tasks/{data}/configs"
    
    configs = os.listdir(config_root) if configs is None else configs
    
    
    command_template = "python main.py --task {task} --data {data} --log {log}"
    
    count = {}
    for config in configs:
        
        
        result_path = os.path.join(config_root,config,"result")
        config = config.replace("(","\(").replace(")","\)")
        
        if os.path.exists(result_path):
            # paths.append(os.path.join(result_path,os.listdir(result_path)[-1]))
            result_files = os.listdir(result_path)
            paths = []
            for result_file in result_files:
                # if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")) \ 
                # and os.path.exists(os.path.join(result_path,result_file,"all")):
                if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")):
                # result_file_path = os.path.join(result_path,result_file,"all")
                # if os.path.exists(result_file_path):
                    paths.append(os.path.join(result_path,result_file))
            for path in paths:
                path = path.replace("(","\(").replace(")","\)")
                command = command_template.format(task = config,
                                                  data = data,
                                                  log = path)
                try:
                    return_val = os.system(command)
                except Exception as e:
                    print(e)
        count[config]=paths
                
                
    # print(count)
    
    # print(len(count))
                    
            
def test_task_logs(data ="PHA_51tenant_5community_28house",
                   ):
    
    config_root = f"LLMGraph/tasks/{data}/configs"
    
    configs = os.listdir(config_root)
  
    not_available_results =[]
    
    for config in configs:
        
        result_path = os.path.join(config_root,config,"result")
        
        if os.path.exists(result_path):
            # paths.append(os.path.join(result_path,os.listdir(result_path)[-1]))
            result_files = os.listdir(result_path)
            paths = []
            ok = False
            for result_file in result_files:
                if os.path.exists(os.path.join(result_path,result_file,"tenental_system.json")):
                    tenental_info = readinfo(os.path.join(result_path,result_file,"tenental_system.json"))
                    last_round = list(tenental_info.keys())[-1]
                    try:
                        if (int(last_round)>=9):
                            ok = True
                    except:
                        pass
            if (not ok):not_available_results.append([config,list(tenental_info.keys())[-1]])
                        
    with open("LLMGraph/tasks/PHA_51tenant_5community_28house/cache/not_available_tasks.json",
              'w',encoding = 'utf-8') as f:
        json.dump(not_available_results, f, indent=4,separators=(',', ':'),ensure_ascii=False)
    
    
def clear_all_cache_ex_data(data):
    task_root_dir = os.path.join("LLMGraph/tasks",data)
    configs = os.listdir(os.path.join(task_root_dir,"configs"))
    for config in configs:
        config_path = os.path.join(task_root_dir,"configs",config)
        result_path = os.path.join(config_path,"result")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
            
def clear_unfinished_ex_data(data):
    task_root_dir = os.path.join("LLMGraph/tasks",data)
    configs = os.listdir(os.path.join(task_root_dir,"configs"))
    for config in configs:
        config_path = os.path.join(task_root_dir,"configs",config)
        result_path = os.path.join(config_path,"result")
        if os.path.exists(result_path):
            results = os.listdir(result_path)
            for result_one in results:
                result_one_path = os.path.join(result_path,result_one)
                if not os.path.exists(os.path.join(result_one_path,"all")):
                    shutil.rmtree(result_one_path)
            results = os.listdir(result_path)
            if len(results) ==0:
                shutil.rmtree(result_path)
            
            
            
def set_data_configs(data):
    task_dir ="LLMGraph/tasks"
    
    config_root = os.path.join(task_dir,data,"configs")
    task_names = os.listdir(config_root)


    # task_names = list(filter(lambda x: 
    #     not os.path.exists(os.path.join(config_root,x,"result")),
    #                          task_names))
    
    dirs = {
        "house":"",
        "tenant":"",
        "forum":"",
        "community":""
    }
    
    distribution_batch_dir={
        "tenant":"",
        "community":""
    }
    
    data_files = os.listdir(os.path.join(task_dir,data,"data"))
    
    data_files = list(filter(lambda x:x!="visualize",data_files))
    
    for data_type  in dirs.keys():
        for data_file in data_files:
            if (data_type in data_file):
                dirs[data_type] = os.path.join("data",data_file)
                break
            
    
    
    for task_name in task_names:
        config_path = os.path.join(config_root,task_name,"config.yaml")
        task_config = yaml.safe_load(open(config_path))
        
        """default k"""
        if task_config["environment"]["rule"]["order"]["type"] == "kwaitlist":
            if "k" not in task_config["environment"]["rule"]["order"].keys():
                task_config["environment"]["rule"]["order"]["k"] = 2
            if "waitlist_ratio" not in task_config["environment"]["rule"]["order"].keys():
                task_config["environment"]["rule"]["order"]["waitlist_ratio"] = 1.2
                
        """communication_num"""
        task_config["environment"]["communication_num"] = 10
        
       
        
        distribution_data_paths = os.listdir(os.path.join(config_root,task_name,"data"))
        for data_path in distribution_data_paths:
            if "tenant" in data_path:
                distribution_batch_dir["tenant"] =  os.path.join("data",data_path)
            else: distribution_batch_dir["community"] = os.path.join("data",data_path)
        
        for data_type,data_dir in dirs.items():
            task_config["managers"][data_type]["data_dir"] = data_dir
        
        for distribution_key,distribution_path in distribution_batch_dir.items():
            task_config["managers"][distribution_key]["distribution_batch_dir"] = distribution_path

            
        task_config["name"] = task_name
        with open(config_path, 'w') as outfile:
            yaml.dump(task_config, outfile, default_flow_style=False)
    
    
def replace_distribution_batch(data):
    task_dir ="LLMGraph/tasks"
    
    config_root = os.path.join(task_dir,data,"configs")
    task_names = os.listdir(config_root)
    
    origin_name = "distribution_batch_28_3.json"
    
    new_name = "distribution_batch_39_3_1.json"
    
    new_json_path = "test/generate_data/distribution_batch_39_3_1.json"
    
    for task_name in task_names:
        if os.path.exists(os.path.join(config_root,task_name,"data",origin_name)):
            origin_file = readinfo(os.path.join(config_root,task_name,"data",origin_name))
            assert len(origin_file)==3
            assert list(origin_file.keys())[1]=="1"
            os.remove(os.path.join(config_root,task_name,"data",origin_name))
            shutil.copyfile(new_json_path,
                            os.path.join(config_root,task_name,"data",new_name))
            
def run_optimizer(optimize_times = 20):
    command_args = [
                "--data","public_housing_optimizer",
                "--optimize",
                "--optimize_regressor_max_samples","60",
                "--optimize_regressor_threshold","0.3",
                "--optimize_refine_first"
            ]   
    
    command_template = "python main.py "
    
    command = command_template + " ".join(command_args) 
    
    for _ in range(optimize_times):
        try:
            return_val = os.system(command)
        except Exception as e:
            print(e)
    
if __name__ == "__main__":
    
    task_dir ="LLMGraph/tasks"

    """multi_list experiment"""
    
    task_names = [
        "ver2_nofilter_multilist(1.2_k1)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.5_k1)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.8_k1)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.5_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.8_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.2_k3)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.5_k3)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3",
        "ver2_nofilter_multilist(1.8_k3)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose3"
    ]
    data = "public_housing"
    
    """entrance policy for house"""
    
    task_names = [
        "ver1_nofilter_singlelist_5t_1h_p#singlelist",
        "ver1_nofilter_singlelist_5t_3h(step_num(t1_h2))_p#singlelist",
        "ver1_nofilter_singlelist_5t_3h(step_num(t1_h2))_p#singlelist",
        "ver1_nofilter_singlelist_5t_3h(step_num(t1_h3))_p#singlelist",
        "ver1_nofilter_singlelist_5t_6h_p#singlelist"
    ]
    data = "public_housing"
   
    """entrance policy for tenant"""
    
    task_names = [
        "ver1_nofilter_singlelist_1t_6h(step_num(t1_h1)_p#singlelist",
        "ver1_nofilter_singlelist_2t_6h(step_num(t1_h1))_p#singlelist",
        "ver1_nofilter_singlelist_2t_6h(step_num(t3_h1))_p#singlelist",
        "ver1_nofilter_singlelist_2t_6h(step_num(t5_h1))_p#singlelist",
        "ver1_nofilter_singlelist_4t_6h(step_num(t1_h1))_p#singlelist",
        "ver1_nofilter_singlelist_4t_6h(step_num(t2_h1))_p#singlelist",
        "ver1_nofilter_singlelist_5t_6h_p#singlelist",
        "ver1_nofilter_singlelist_8t_6h_p#singlelist"
    ]
    data = "public_housing"
    
    """allocation experiments"""
     
    task_names =[
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#random_avg",
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose2",
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#portion_rentmoney",
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#random_avg",
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#housetype",
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#portion_rentmoney",
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_portion3(f_member_num)_priority_8t_6h_p#random_avg",
        "ver1_nofilter_multilist(1.2)_portion3(f_member_num)_priority_8t_6h_p#portion_rent_money",
        "ver1_nofilter_multilist(1.2)_portion3(f_member_num)_priority_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_portion3(f_rent_money_budget)_priority_8t_6h_p#random_avg",
        "ver1_nofilter_multilist(1.2)_portion3(f_rent_money_budget)_priority_8t_6h_p#portion_rent_money",
        "ver1_nofilter_multilist(1.2)_portion3(f_rent_money_budget)_priority_8t_6h_p#portion_housesize"
    ]
    data = "public_housing"
    
    """sorting policy"""
    data = "public_housing_sorting_policy"
    task_names =[
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#housetype",
        "ver1_nofilter_multilist(1.2)_multilist_housing_points_8t_6h_p#housetype",
        "ver2_nofilter_multilist(1.2_k2)_housetype_priority_8t_6h(step_num(t1_h1))_p#housetype_choose2",
        "ver2_nofilter_multilist(1.2_k2)_housetype_housing_points_8t_6h(step_num(t1_h1))_p#housetype_choose2",
        "ver1_nofilter_multilist(1.2)_portion3(f_earn_money)_priority_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_portion3(f_earn_money)_housing_points_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_multilist_priority_8t_6h_p#portion_rentmoney",
        "ver1_nofilter_multilist(1.2)_multilist_housing_points_8t_6h_p#portion_rentmoney",
        "ver1_nofilter_multilist(1.2)_multilist_nopriority_8t_6h_p#housetype",
        "ver1_nofilter_multilist(1.2)_multilist_nopriority_8t_6h_p#housetype",
        "ver2_nofilter_multilist(1.2_k2)_housetype_nopriority_8t_6h(step_num(t1_h1))_p#housetype_choose2",
        "ver2_nofilter_multilist(1.2_k2)_housetype_nopriority_8t_6h(step_num(t1_h1))_p#housetype_choose2",
        "ver1_nofilter_multilist(1.2)_portion3(f_earn_money)_nopriority_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_portion3(f_earn_money)_nopriority_8t_6h_p#portion_housesize",
        "ver1_nofilter_multilist(1.2)_multilist_nopriority_8t_6h_p#portion_rentmoney",
        "ver1_nofilter_multilist(1.2)_multilist_nopriority_8t_6h_p#portion_rentmoney"
    ]
    
    
    log_dir = f"LLMGraph/tasks/{data}/cache"
    

    
    run_tasks(task_names,
              data,
              log_dir,
              run_ex_times=2)
    
    # replace_distribution_batch(data)
    
    # set_data_configs(data)
    
    """ 谨慎执行，有可能导致未结束的实验 进行matrix计算 """
    # run_tasks_logs(data,
    #                configs=task_names)
    
    # test_task_logs()
    
    
    """ 请谨慎执行，确保备份 """
    # clear_all_cache_ex_data(data)
    
    # clear_unfinished_ex_data(data)