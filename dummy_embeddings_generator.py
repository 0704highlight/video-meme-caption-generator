import numpy as np
import json
import os

# 嵌入向量维度
EMBEDDING_DIM = 128

def main():
    # 确保database目录存在
    os.makedirs('database', exist_ok=True)
    
    # 加载examples.json
    try:
        with open('database/examples.json', 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except Exception as e:
        print(f"加载examples.json时出错: {e}")
        return
    
    # 生成随机嵌入向量
    num_examples = len(examples)
    embeddings = np.random.random((num_examples, EMBEDDING_DIM))
    
    # 保存嵌入向量
    try:
        np.save('database/embeddings.npy', embeddings)
        print(f"成功生成{num_examples}个嵌入向量，并保存到database/embeddings.npy")
    except Exception as e:
        print(f"保存嵌入向量时出错: {e}")

if __name__ == "__main__":
    main() 