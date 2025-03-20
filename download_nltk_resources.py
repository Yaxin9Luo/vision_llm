#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载NLTK所需的资源
运行此脚本以确保所有必要的NLTK资源都已下载
"""

import nltk
import ssl
import sys

def download_nltk_resources():
    """下载NLTK分析所需的资源"""
    
    # 尝试修复SSL证书问题（如果有）
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # 要下载的资源列表
    resources = [
        'punkt',           # 用于句子分割
        'stopwords',       # 停用词
        'vader_lexicon',   # 情感分析
        'wordnet',         # 词义分析
        'omw-1.4',         # Open Multilingual WordNet
    ]
    
    # 下载每个资源
    for resource in resources:
        try:
            print(f"正在下载 {resource}...")
            nltk.download(resource)
            print(f"{resource} 下载成功")
        except Exception as e:
            print(f"下载 {resource} 时出错: {e}")
    
    # 测试资源是否可用
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # 测试分词
        test_text = "这是一个测试句子。这是另一个句子。"
        sentences = sent_tokenize(test_text)
        words = word_tokenize(sentences[0])
        print(f"分词测试 - 句子数: {len(sentences)}, 第一个句子的词数: {len(words)}")
        
        # 测试停用词
        stop_words = set(stopwords.words('english'))
        print(f"停用词测试 - 停用词数量: {len(stop_words)}")
        
        # 测试情感分析
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores("I love this!")
        print(f"情感分析测试 - 积极情感分数: {sentiment['pos']}")
        
        print("\n所有测试通过！NLTK资源已成功下载并可用。")
        
    except Exception as e:
        print(f"测试NLTK资源时出错: {e}")
        print("请尝试手动下载所有资源: nltk.download('all')")
        return False
    
    return True

if __name__ == "__main__":
    print("开始下载NLTK资源...")
    success = download_nltk_resources()
    
    if success:
        print("\n所有NLTK资源已成功下载。您现在可以运行可视化分析脚本了。")
        sys.exit(0)
    else:
        print("\n下载NLTK资源时出现问题。请尝试手动下载。")
        sys.exit(1) 