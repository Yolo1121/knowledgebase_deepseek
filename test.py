from pymilvus import connections, Collection, utility
import numpy as np

def check_collection_details(collection_name="sjznNew"):
    try:
        # 连接到 Milvus
        print(f"连接到 Milvus 并检查集合 {collection_name}...")
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        
        if collection_name not in utility.list_collections():
            print(f"集合 {collection_name} 不存在!")
            return
            
        collection = Collection(collection_name)
        collection.load()
        
        # 添加这些调试信息
        print("\n集合信息:")
        print(f"- 名称: {collection_name}")
        print(f"- 记录数: {collection.num_entities}")
        print(f"- Schema: {collection.schema}")
        
        # 获取实际的字段名
        field_names = [field.name for field in collection.schema.fields]
        print(f"\n可用字段: {field_names}")
        
        # 使用实际存在的字段进行查询
        results = collection.query(
            expr="",
            output_fields=field_names,  # 使用所有可用字段
            limit=5
        )
        
        print("\n前5条记录:")
        for i, result in enumerate(results, 1):
            print(f"\n记录 {i}:")
            print(result)
        
        # 显示集合统计信息
        print(f"\n集合 {collection_name} 的详细信息:")
        print(f"- 总记录数: {collection.num_entities}")
        
        # 显示索引信息
        index_info = collection.index().params
        print("\n索引信息:")
        print(f"- 索引类型: {index_info.get('index_type', 'unknown')}")
        print(f"- 度量类型: {index_info.get('metric_type', 'unknown')}")
        
        # 显示示例数据
        print("\n数据示例:")
        sample_results = collection.query(
            expr="",
            output_fields=field_names,  # 使用之前获取的实际字段名
            limit=2
        )
        
        for i, result in enumerate(sample_results, 1):
            print(f"\n记录 {i}:")
            for key, value in result.items():
                if key == "vector":
                    vector = np.array(value)
                    print(f"- 向量信息:")
                    print(f"  • 维度: {vector.shape}")
                    print(f"  • 前5个值: {vector[:5]}")
                    print(f"  • L2范数: {np.linalg.norm(vector)}")
                else:
                    print(f"- {key}: {value}")
        
    except Exception as e:
        print(f"检查集合时出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            connections.disconnect("default")
            print("\n已断开 Milvus 连接")
        except:
            pass

if __name__ == "__main__":
    check_collection_details()
