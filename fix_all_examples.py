#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一次性运行的脚本，用于处理examples.json中的所有情感标签，
使用与agent.py中相同的方法提取关键词
"""

import os
import json
import shutil
from config import DB_PATH

def fix_all_emotion_tags():
    """修复examples.json中的所有情感标签"""
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
        
        # 定义常见情感关键词列表
        common_emotions = ['开心', '高兴', '快乐', '兴奋', '喜悦', '愉快', 
                          '悲伤', '难过', '忧郁', '伤心', '沮丧', '痛苦',
                          '愤怒', '生气', '恼火', '暴怒', '不满', '焦虑',
                          '害怕', '恐惧', '担忧', '紧张', '惊讶', '震惊',
                          '惊喜', '惊吓', '惊恐', '困惑', '迷茫', '疑惑', '好奇',
                          '厌恶', '鄙视', '嫌弃', '讨厌', '无聊', '无奈',
                          '尴尬', '羞愧', '害羞', '自卑', '自豪', '自信',
                          '满足', '满意', '轻松', '舒适', '平静', '安心',
                          '委屈', '泪水', '流泪', '哭泣', '微笑', '温和']
        
        long_tags_count = 0
        fixed_examples = 0
        
        # 处理每个示例
        for example in examples:
            if 'emotion_tags' in example and example['emotion_tags']:
                original_tags = example['emotion_tags'].copy()
                processed_tags = []
                has_long_tag = False
                
                for tag in original_tags:
                    if len(tag) > 10:  # 长句标签
                        has_long_tag = True
                        long_tags_count += 1
                        print(f"发现长句标签: '{tag}'")
                        
                        # 从句子中提取情感关键词
                        found_keywords = []
                        for emotion in common_emotions:
                            if emotion in tag and emotion not in found_keywords:
                                found_keywords.append(emotion)
                                print(f"  - 提取到关键词: '{emotion}'")
                        
                        # 如果找到了关键词，使用它们
                        if found_keywords:
                            processed_tags.extend(found_keywords)
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
                
                # 如果有长句标签，则更新示例
                if has_long_tag:
                    fixed_examples += 1
                    example['emotion_tags'] = processed_tags
                    print(f"情感标签变更:")
                    print(f"  - 原始标签: {original_tags}")
                    print(f"  - 处理后标签: {processed_tags}")
        
        # 保存修复后的示例数据
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        print(f"\n修复完成！")
        print(f"处理了 {len(examples)} 个示例")
        print(f"发现 {long_tags_count} 个长句标签")
        print(f"修复了 {fixed_examples} 个示例的情感标签")
        
        # 删除旧的FAISS索引，强制下次运行时重建
        if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
            shutil.rmtree(faiss_index_path)
            print(f"已删除旧的FAISS索引，下次运行时将自动重建")
        
        return True
        
    except Exception as e:
        print(f"修复情感标签时出错: {e}")
        # 如果出错，尝试恢复备份
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, examples_path)
            print(f"已恢复原始文件")
        return False

if __name__ == "__main__":
    print("开始修复examples.json文件中的情感标签...")
    success = fix_all_emotion_tags()
    if success:
        print("修复成功！")
    else:
        print("修复失败！") 