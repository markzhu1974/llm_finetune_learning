from rouge_chinese import Rouge
from jinja2 import Template
import jieba # you can use any other word cutting library
import re
import time
import json
import requests
import numpy as np
from tqdm import tqdm


def equal_eval(hypothesis, reference):
    """判断两个句子是否完全相同，返回评估值
    """
    count = len(hypothesis)
    right = 0
    for sent, gold in zip(hypothesis, reference):
        if sent == gold:
            right += 1
    
    return [{"name": "right", "value": right, "description": "生成答案和标准答案完全相同的数量"},
            {"name": "count", "value": count, "description": "样本总数量"},
            {"name": "acc", "value": right / count, "description": "生成答案和标准答案完全相同的比例"}]

def cover_eval(hypothesis, reference):
    """判断生成答案是否包含标准答案中的所有内容，返回评估值"""
    count = len(hypothesis)
    right = 0
    for sent, gold in zip(hypothesis, reference):
        if gold in sent:
            right += 1
    
    return [{"name": "right", "value": right, "description": "生成答案和标准答案完全相同的数量"},
            {"name": "count", "value": count, "description": "样本总数量"},
            {"name": "acc", "value": right / count, "description": "生成答案包含标准答案的比例"}]


def rouge_eval(hypothesis, reference):
    """计算rouge-1, rouge-2, rouge-l"""
    count = len(hypothesis)
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    rouge = Rouge()
    for sent, gold in zip(hypothesis, reference):
        sent = ' '.join(jieba.cut(sent))
        gold = ' '.join(jieba.cut(gold)) 
        scores = rouge.get_scores(sent, gold)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
    
    return [
        {"name": "rouge-1", "value": rouge_1 / count, "description": "rouge-1 指标平均得分"},
        {"name": "rouge-2", "value": rouge_2 / count, "description": "rouge-2 指标平均得分"},
        {"name": "rouge-l", "value": rouge_l / count, "description": "rouge-l 指标平均得分"}
    ]

def cosine_similarity(vector1, vector2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
