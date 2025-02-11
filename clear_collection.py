from pymilvus import connections, utility

def clear_collection(collection_name: str):
    try:
        # 连接到 Milvus
        connections.connect(
            alias="default",
            host="localhost",  # 改用本地地址
            port=19530
        )
        
        # 删除集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"成功删除集合: {collection_name}")
        
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f"清空集合时出错: {str(e)}")
        return False

if __name__ == "__main__":
    collection_name = "sjznNew"  # 替换为你的集合名
    clear_collection(collection_name)