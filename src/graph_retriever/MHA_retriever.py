
import argparse
import yaml
from typing import Callable, Dict, List, Tuple
import json
import pandas as pd
import os

# 核心依赖：复用 evaluate.py 和其他模块中的类和函数
from multirag.storage import VectorDB, DistanceMetric
from multirag.embed import load_embeddings, QueryEmbeddings
from multirag.dataset import Article
from multirag.evaluation import StandardStrategy, MultiHeadStrategy, Strategy

def get_db(db_config_path: str) -> VectorDB:

    print(f"Connecting to the database using config: {db_config_path}...")

    with open(db_config_path, 'r') as file:
        docker_config = yaml.safe_load(file)
    container_config = docker_config['services']['postgres']
    db_config: dict[str, str] = {}
    for param in container_config['environment']:
        key, value = param.split('=')
        db_config[key] = value

    return VectorDB(
        distance_metric=DistanceMetric.COSINE,
        port=5428,
        name=db_config['POSTGRES_DB'],
        user=db_config['POSTGRES_USER'],
        password=db_config['POSTGRES_PASSWORD']
    )

# --- 核心检索函数 ---

def retrieve_top_k(
    query_embedding: QueryEmbeddings,
    k: int,
    layer: int,
    db: VectorDB,
    query_id: int
) -> Dict[str, Tuple[Article, ...]]:
    """
    为一个给定的查询嵌入，使用多种策略检索 Top-K 的文章。

    :param query_embedding: 包含查询文本、主题和嵌入的对象。
    :param k: 需要检索的文章数量。
    :param layer: MultiHeadStrategy 要使用的层。
    :param db_config_path: 数据库 docker-compose 配置文件的路径。
    :return: 一个字典，键是策略名称，值是检索到的 Article 对象元组。
    """
    

    # 2. 初始化要使用的策略
    strategies: List[Strategy] = [
        StandardStrategy("standard-rag", db),
        MultiHeadStrategy("multirag", db, layer, lambda h, r, d: h * (2 ** -r)),
    ]

    # 3. 执行检索并收集结果
    results: Dict[str, Tuple[Article, ...]] = {}
    for strategy in strategies:
        retrieved_entities = strategy._get_entity_picks(query_embedding, k,query_id)
        retrieved_relations = strategy._get_relation_picks(query_embedding, k,query_id)
        results[strategy.name+'_entity'] = retrieved_entities
        results[strategy.name+'_relation'] = retrieved_relations

    # 4. 返回结果
    return results

# --- 主执行流程 ---

def build_gid_mappings(nodes_dir: str, edges_dir: str, idx: str):
    """
    为给定的子图 id，加载 nodes/{id}.csv 和 edges/{id}.csv，
    返回两个映射 dict：
      gid2nid: gid_str -> node_id (int)
      gid2eid: gid_str -> edge_index (int)
    """
    nodes_df = pd.read_csv(os.path.join(nodes_dir, f"{idx}.csv"), dtype=str)
    edges_df = pd.read_csv(os.path.join(edges_dir, f"{idx}.csv"), dtype=str)

    gid2nid = {}
    for _, row in nodes_df.iterrows():
        raw_gid = row["gid"]                # 如 "4569" 或者可能已经是 "n4569"
        nid     = int(row["node_id"])
        # 1) 把原始的 raw_gid 映射到 nid
        gid2nid[raw_gid] = nid
        # 2) 如果没前缀，就同时支持带前缀的版本
        if not raw_gid.startswith("n"):
            gid2nid["n" + raw_gid] = nid
        # 3) 如果带了前缀，也保留不带前缀的版本
        else:
            gid2nid[raw_gid.lstrip("n")] = nid

    gid2eid = {}
    for i, (_, row) in enumerate(edges_df.iterrows()):
        raw_gid = row["gid"]  # 如 "20" 或 "e20"
        # 同理做双向映射
        gid2eid[raw_gid] = i
        if not raw_gid.startswith("e"):
            gid2eid["e" + raw_gid] = i
        else:
            gid2eid[raw_gid.lstrip("e")] = i

    return gid2nid, gid2eid

def main():
    parser = argparse.ArgumentParser(
    description="Batch retrieve and output node_id/edge_id for each strategy"
    )
    # 1. 核心参数
    parser.add_argument('-c', '--db-config', type=str, required=True,
                        help='Path to docker-compose.yaml')
    parser.add_argument('-d', '--data-path', type=str, required=True,
                        help='Base directory for all data (inputs and outputs)')

    # 2. 其他配置参数
    parser.add_argument('--k', type=int, default=5,
                        help='Top-K per query')
    parser.add_argument('--layer', type=int, default=31,
                        help='Layer for MultiHeadStrategy')

    # 3. 将 -o 参数改为可选的、只指定【文件名】的参数
    parser.add_argument('--output-filename', type=str, default='mha_retrieve.jsonl',
                        help='Filename for the output JSONL file, will be saved inside the data-path directory.')

    args = parser.parse_args()

    # --- 在代码中构建所有路径 ---

    # 4. 从基准路径构建输入文件/目录的路径
    embedding_path = os.path.join(args.data_path, 'embeddings.json')
    nodes_dir = os.path.join(args.data_path, 'nodes')
    edges_dir = os.path.join(args.data_path, 'edges')

    # 5. 从基准路径和输出文件名构建【输出文件】的完整路径
    output_path = os.path.join(args.data_path, args.output_filename)

    # 1. load embeddings & DB
    print("Loading embeddings...")
    _, query_embeddings = load_embeddings(embedding_path)
    print("Connecting to DB...")
    db = get_db(args.db_config)

    fout = open(output_path, 'w', encoding='utf-8')
    for idx, q_emb in enumerate(query_embeddings):
        idx_str = str(idx)  # 与 CSV 同名的子图 id
        print(f"[{idx_str}] processing...")

        # 2. build gid->id mappings for this subgraph
        gid2nid, gid2eid = build_gid_mappings(
            nodes_dir, edges_dir, idx_str
        )

        # 3. retrieve
        results = retrieve_top_k(
            query_embedding=q_emb,
            k=args.k,
            layer=args.layer,
            db=db,
            query_id=idx
        )

        # 4. convert each strategy's GID-list into id-list
        converted = {}
        for strat, arts in results.items():
            # strat is like "standard-rag_entity" or "..._relation"
            if strat.endswith("_entity"):
                # arts: tuple of Article, title==gid
                converted[strat] = [
                    gid2nid.get(art.title, None) for art in arts
                ]
            else:  # relation
                converted[strat] = [
                    gid2eid.get(art.title, None) for art in arts
                ]

        # 5. write JSONL
        rec = {
            "query_index": idx,
            "query_text": q_emb.query.text,
            "results": converted
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fout.close()
    print("Done! Written to", output_path)


if __name__ == "__main__":
    main()