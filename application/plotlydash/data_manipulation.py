import numpy as np
import ast
import plotly.express as px
import pandas as pd
import  json

# FILE_PATH = '/Users/wenchenliu/Desktop/dpt/cleaned_BRFSS.csv'
FILE_PATH = '/Users/wenchenliu/Desktop/Interactive-Machine-Learning-Tool-for-Risk-Factor-Analysis/download.csv'
df = pd.read_csv(FILE_PATH)

file = open('/Users/wenchenliu/Desktop/Interactive-Machine-Learning-Tool-for-Risk-Factor-Analysis/data/var_info.txt', 'r')
contents = file.read()
dictionary = ast.literal_eval(contents)
file. close()

#return a set of all section names
def section_name_dictionary(dict):
    name_set = set()

    for keys in dict:
        name = dict[keys]["Section Name"]
        name_set.add(name)
    return name_set

# print(section_name_dictionary(dictionary))


# return a dictionary where the key is the section name and the values are
# a list of all the features that share the same section name
def find_correspond_feature(set,dict):
    sec_name_list = list(set)
    sec_dictionary = {}
    for names in sec_name_list:
        append_list = []
        for keys in dict:
            if dict[keys]["Section Name"] == names:
                append_list.append(keys)
        sec_dictionary[names] = append_list
    return sec_dictionary

# return a list of lists where the first list are the possible categorical values and second
# list can only contain non categorical values
def categorize_features(df):
    floats = []
    integers = []

    for i in df.columns:
        integer_holder = []
        col_items = df[i].tolist()
        for x in col_items:
            if isinstance(x,float):
                if x.is_integer():
                    integer_holder.append(x)
            elif isinstance(x,int):
                integer_holder.append(i)

        if len(col_items) == len(integer_holder):
            integers.append(i)
        else:
             floats.append(i)
    result = [integers, floats]
    return result
#
# print(categorize_features(df))
#
m = categorize_features(df)
print(len(m))
with open('/Users/wenchenliu/Desktop/Interactive-Machine-Learning-Tool-for-Risk-Factor-Analysis/data/categorized_type.txt', 'w') as file1:
    file1.write(json.dumps(m))



#_-_______________________________
# file2 = open('data/categorized_type', 'r')
# contents2 = file2.read()
# categories = ast.literal_eval(contents2)
# file2.close()
#
# categories


# print(df)
#
# print(len(df['_STATE'].tolist()))


# 364360    56.0
# m = find_correspond_feature(section_name_dictionary(dictionary), dictionary)
# print(len(m))
# with open('/Users/wenchenliu/Desktop/Interactive-Machine-Learning-Tool-for-Risk-Factor-Analysis/data/section_name.txt', 'w') as file1:
#     file1.write(json.dumps(m))