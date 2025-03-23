#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理examples.json文件，移除anime_tags字段
并重建FAISS索引
"""

import os
import json
import shutil
from config import DB_PATH

def clean_examples_file():
    """清理examples.json文件中的anime_tags字段"""
    examples_path = os.path.join(DB_PATH, "examples.json")
    faiss_index_path = os.path.join(DB_PATH, "faiss_index")
    
    if not os.path.exists(examples_path):
        print(f"示例文件不存在: {examples_path}")
        return False
    
    # 备份原始文件
    backup_path = examples_path + ".bak"
    shutil.copy2(examples_path, backup_path)
    print(f"已备份原始文件到: {backup_path}")
    
    # 读取示例数据
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        print(f"原始示例数量: {len(examples)}")
        
        # 移除每个示例中的anime_tags字段
        cleaned_examples = []
        anime_tags_count = 0
        
        for example in examples:
            if 'anime_tags' in example:
                anime_tags_count += 1
                del example['anime_tags']
            cleaned_examples.append(example)
        
        print(f"移除了 {anime_tags_count} 个anime_tags字段")
        
        # 保存清理后的数据
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_examples, f, ensure_ascii=False, indent=2)
        
        print(f"已保存清理后的示例数据")
        
        # 删除旧的FAISS索引，强制下次运行时重建
        if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
            shutil.rmtree(faiss_index_path)
            print(f"已删除旧的FAISS索引，下次运行时将自动重建")
        
        return True
    
    except Exception as e:
        print(f"清理示例数据时出错: {e}")
        # 如果出错，尝试恢复备份
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, examples_path)
            print(f"已恢复原始文件")
        return False

def clean_emotion_tags():
    file_path = './database/examples.json'
    backup_path = file_path + '.bak'
    
    # 创建备份
    shutil.copy2(file_path, backup_path)
    print(f"已备份原始文件到: {backup_path}")
    
    # 读取JSON数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_count = 0
    
    # 常见情感关键词列表
    common_emotions = ['开心', '高兴', '快乐', '兴奋', '喜悦', '愉快', 
                      '悲伤', '难过', '忧郁', '伤心', '沮丧', '痛苦',
                      '愤怒', '生气', '恼火', '暴怒', '不满', '焦虑',
                      '害怕', '恐惧', '担忧', '紧张', '惊讶', '震惊',
                      '惊喜', '惊吓', '惊恐', '困惑', '迷茫', '疑惑', '好奇',
                      '厌恶', '鄙视', '嫌弃', '讨厌', '无聊', '无奈',
                      '尴尬', '羞愧', '害羞', '自卑', '自豪', '自信',
                      '满足', '满意', '轻松', '舒适', '平静', '安心',
                      '委屈', '泪水', '流泪', '哭泣', '微笑', '温和']
    
    # 打印情感关键词列表，便于调试
    print(f"情感关键词列表: {', '.join(common_emotions)}")
    print("开始处理情感标签...")
    
    # 处理每条数据
    for item in data:
        if 'emotion_tags' in item:
            original_tags = item['emotion_tags'].copy() if isinstance(item['emotion_tags'], list) else []
            processed_tags = []
            
            has_long_tag = False
            for tag in item['emotion_tags']:
                # 如果标签很长，可能是一个描述句子
                if len(tag) > 10:
                    has_long_tag = True
                    cleaned_count += 1
                    print(f"发现长句标签: '{tag}'")
                    
                    # 从句子中提取情感关键词
                    found_emotions = []
                    for emotion in common_emotions:
                        if emotion in tag and emotion not in found_emotions:
                            found_emotions.append(emotion)
                            print(f"  - 提取关键词: '{emotion}'")
                    
                    # 如果找到了关键词，使用它们
                    if found_emotions:
                        processed_tags.extend(found_emotions)
                    else:
                        print(f"  - 未找到匹配的情感关键词，尝试基本处理")
                        # 如果没找到，尝试使用更基本的处理
                        words = tag.replace('，', ',').replace('。', '').replace('、', ',').split(',')
                        for word in words:
                            if len(word) <= 4 and len(word) > 0:  # 短词更可能是情感词
                                clean_word = word.strip()
                                processed_tags.append(clean_word)
                                print(f"  - 提取短词: '{clean_word}'")
                else:
                    # 短标签直接使用
                    processed_tags.append(tag)
            
            # 只有在有长句标签时才打印变更信息
            if has_long_tag:
                print(f"情感标签变更:")
                print(f"  - 原始标签: {original_tags}")
                print(f"  - 处理后标签: {processed_tags}")
            
            # 更新情感标签
            item['emotion_tags'] = processed_tags
    
    # 保存清理后的数据
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n清理完成！原始示例数: {original_count}, 清理了 {cleaned_count} 个情感标签。")
    print(f"原始文件已备份到 {backup_path}")
    print("清理后的数据已保存。")

if __name__ == "__main__":
    print("开始清理examples.json文件...")
    success = clean_examples_file()
    if success:
        print("清理完成！")
    else:
        print("清理失败！")

    clean_emotion_tags() 