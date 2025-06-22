#!/usr/bin/env python3
import os
import argparse
import yaml
import pandas as pd
from multirag.storage import VectorDB, DistanceMetric


def get_db(db_config_path: str) -> VectorDB:
    """Connect to PostgreSQL via VectorDB (multirag.storage)."""
    print(f"Connecting to the database using config: {db_config_path}...")
    with open(db_config_path, 'r') as file:
        docker_config = yaml.safe_load(file)
    container_config = docker_config['services']['postgres']
    db_config = {}
    for param in container_config['environment']:
        key, value = param.split('=', 1)
        db_config[key] = value

    return VectorDB(
        distance_metric=DistanceMetric.COSINE,
        port=5428,
        name=db_config['POSTGRES_DB'],
        user=db_config['POSTGRES_USER'],
        password=db_config['POSTGRES_PASSWORD']
    )


def store_entities(db: VectorDB, entities_txt: str):
    """
    Read entities_txt (one node_attr per line), assign gid = line_index (0-based),
    and insert into entities(node_attr, gid).
    """
    conn = db._conn
    cur = conn.cursor()
    with open(entities_txt, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            attr = line.strip()
            if not attr:
                continue
            gid = 'n' + str(idx)
            # Upsert: avoid duplicates
            cur.execute(
                """
                INSERT INTO entities (node_attr, gid)
                VALUES (%s, %s)
                ON CONFLICT (node_attr) DO UPDATE
                  SET gid = EXCLUDED.gid;
                """,
                (attr, gid)
            )
    conn.commit()
    print(f"Stored {idx+1} entities from {entities_txt}.")


def store_edges(db: VectorDB, edges_txt: str):
    """
    Read edges_txt (one edge_attr per line), assign gid = line_index (0-based),
    and insert into edges(edge_attr, gid).
    """
    conn = db._conn
    cur = conn.cursor()
    with open(edges_txt, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            attr = line.strip()
            if not attr:
                continue
            gid = 'e' + str(idx)
            cur.execute(
                """
                INSERT INTO edges (edge_attr, gid)
                VALUES (%s, %s)
                ON CONFLICT (edge_attr) DO UPDATE
                  SET gid = EXCLUDED.gid;
                """,
                (attr, gid)
            )
    conn.commit()
    print(f"Stored {idx+1} edges from {edges_txt}.")


def store_graphs_and_triples(db: VectorDB, nodes_dir: str, edges_dir: str):
    """
    For each edges CSV in edges_dir, create a graph entry and its triples.
    Filename (without .csv) is question_id.
    Nodes CSV with same name in nodes_dir provides node_id->gid mapping.
    """
    conn = db._conn
    cur = conn.cursor()
    for fname in sorted(os.listdir(edges_dir)):
        if not fname.endswith('.csv'):
            continue
        qid = os.path.splitext(fname)[0]
        # Insert into graphs
        cur.execute(
            "INSERT INTO graphs (name, question_id) VALUES (%s, %s) RETURNING id",
            (f'graph_{qid}', qid)
        )
        graph_id = cur.fetchone()[0]
        # Load node mapping
        nodes_path = os.path.join(nodes_dir, f"{qid}.csv")
        nodes_df = pd.read_csv(nodes_path)
        node_map = dict(zip(nodes_df['node_id'], nodes_df['gid']))
        # Load edges and insert triples
        edges_path = os.path.join(edges_dir, fname)
        edges_df = pd.read_csv(edges_path)
        for _, row in edges_df.iterrows():
            head_gid = node_map.get(row['src'])
            tail_gid = node_map.get(row['dst'])
            edge_gid = row['gid']
            if head_gid is None or tail_gid is None:
                print(f"Warning: missing node gid for {qid}: {row}")
                continue
            cur.execute(
                "INSERT INTO graph_triples (graph_id, head_gid, edge_gid, tail_gid) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (graph_id, 'n'+ str(head_gid), 'e' + str(edge_gid),'n' + str(tail_gid))
            )
        conn.commit()
        print(f"Stored graph {qid} with id={graph_id}, {len(edges_df)} triples.")


def main():
    # <<< MODIFIED: 简化命令行参数 >>>
    parser = argparse.ArgumentParser(description="Store entities and edges into DB from a single data directory.")
    parser.add_argument('-c', '--db-config', type=str, required=True,
                        help='Path to docker-compose YAML for postgres env')
    # 将四个路径参数合并成一个
    parser.add_argument('-d', '--data-path', type=str, required=True,
                        help='Base directory containing entities.txt, edges.txt, nodes/, and edges/ directories')
    args = parser.parse_args()
    base_path = args.data_path

    # 获取数据库连接
    db = get_db(args.db_config)
    
    # <<< MODIFIED: 调用辅助函数时，只传递 data_path >>>
    entities_file_path = os.path.join(base_path, 'entities.txt')
    store_entities(db, entities_file_path)

    # 2. 为 store_edges 构建完整的 'edges.txt' 文件路径
    edges_file_path = os.path.join(base_path, 'edges.txt')
    store_edges(db, edges_file_path)

    # 3. 为 store_graphs_and_triples 构建 'nodes' 和 'edges' 目录的路径
    nodes_dir_path = os.path.join(base_path, 'nodes')
    edges_dir_path = os.path.join(base_path, 'edges')
    store_graphs_and_triples(db, nodes_dir_path, edges_dir_path)
    # >>> END MODIFICATION

    print("✅ All tasks have been completed successfully.")



if __name__ == '__main__':
    main()
