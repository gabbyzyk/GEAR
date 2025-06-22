import yaml
import torch
import argparse
import torch.nn.functional as F
from torch_geometric.transforms import GDC
from torch_geometric.data import Data
from pcst_fast import pcst_fast
import pandas as pd
import numba
import numpy as np
import json
import os
import io





def build_desc_from_indices_standard(graph, topk_n_indices, topk_e_indices, textual_nodes, textual_edges):
    """
    根据预先计算好的top-k节点和边的索引，构建子图的文本描述。
    此版本仅使用图的结构信息，不关心节点的特征向量。

    Args:
        graph (Data): 原始的PyTorch Geometric图对象，主要使用其 .edge_index。
        topk_n_indices (Tensor or list): 预先选出的高分节点的索引。
        topk_e_indices (Tensor or list): 预先选出的高分边的索引。
        textual_nodes (DataFrame): 节点的文本描述。
        textual_edges (DataFrame): 边的文本描述。

    Returns:
        str: 描述子图的CSV格式字符串。
    """
    
    # 边界条件处理：如果文本描述为空，直接返回一个空的描述字符串。
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        return textual_nodes.to_csv(index=False) + '\n' + \
               textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    # --- 核心逻辑开始 ---

    # 确保输入的索引是Tensor格式，以便进行后续的torch操作
    device = graph.edge_index.device
    if not isinstance(topk_n_indices, torch.Tensor):
        topk_n_indices = torch.tensor(topk_n_indices, dtype=torch.long, device=device)
    if not isinstance(topk_e_indices, torch.Tensor):
        topk_e_indices = torch.tensor(topk_e_indices, dtype=torch.long, device=device)

    # 1. 节点合并与扩展：确定子图包含的所有节点
    # 这一步是为了确保我们描述的子图是连贯的。
    
    row, col = graph.edge_index
    # 从被选中的高分边中，提取出它们所连接的所有节点。
    selected_edge_nodes = torch.cat([row[topk_e_indices], col[topk_e_indices]]).unique()
    
    # 将直接提供的高分节点和高分边所连接的节点合并，得到最终的节点全集。
    selected_nodes = torch.unique(torch.cat([topk_n_indices, selected_edge_nodes]))
    
    # 2. 确定子图包含的所有边
    # 筛选出所有两个端点都在最终节点集中的边，以构建一个完整的导出子图描述。
    selected_nodes_set = set(selected_nodes.tolist())
    
    final_edge_indices = []
    for i in range(graph.edge_index.shape[1]):
        u = graph.edge_index[0, i].item()
        v = graph.edge_index[1, i].item()
        if u in selected_nodes_set and v in selected_nodes_set:
            final_edge_indices.append(i)
    # 将最终的边索引列表转换为Tensor
    final_edge_indices = torch.tensor(final_edge_indices, dtype=torch.long, device=device)
    
    # 3. 生成最终的文本描述
    # 使用 .iloc 从DataFrame中按整数位置选择行
    # 需要先将tensor转为numpy数组
    nodes_df = textual_nodes.iloc[selected_nodes.cpu().numpy()]
    edges_df = textual_edges.iloc[final_edge_indices.cpu().numpy()]

    # 将DataFrame转换为CSV格式的字符串
    node_map = pd.Series(nodes_df.node_attr.values, index=nodes_df.node_id).to_dict()

    triplets = []
    # 2. 遍历每一条边
    for _, edge in edges_df.iterrows():
        # 3. 使用映射查找源节点和目标节点的文本属性
        src_attr = node_map.get(edge['src'])
        dst_attr = node_map.get(edge['dst'])
        edge_attr = edge['edge_attr']

        # 确保所有部分都找到了
        if src_attr is not None and dst_attr is not None:
            triplets.append((src_attr, edge_attr, dst_attr))
    
    return triplets












def prr_from_topk(graph, topk_n_indices, topk_e_indices,
                                         textual_nodes, textual_edges,
                                         topk_final=30, alpha=0.15):
    print("--- 正在运行: 基于 topk 节点+边 的 PPR 子图构造算法 ---")

    device = graph.edge_index.device

    # === 步骤1：构建种子节点集合（来自 topk node 和 topk edge）===
    if not isinstance(topk_n_indices, torch.Tensor):
        topk_n_indices = torch.tensor(topk_n_indices, dtype=torch.long, device=device)
    if not isinstance(topk_e_indices, torch.Tensor):
        topk_e_indices = torch.tensor(topk_e_indices, dtype=torch.long, device=device)

    # 从边索引中提取连接的节点对
    edge_nodes = graph.edge_index[:, topk_e_indices]
    edge_node_set = torch.unique(edge_nodes)

    # 合并并去重：topk节点 + topk边涉及节点
    seed_node_indices = torch.unique(torch.cat([topk_n_indices, edge_node_set]))

    if len(seed_node_indices) == 0:
        raise ValueError("种子节点集合为空，无法执行 PPR。")

    # 自动调整topk_final（避免返回太少节点）
    topk_final = max(topk_final, len(seed_node_indices) * 3)

    # === 步骤2：运行 GDC（个性化 PageRank）===
    gdc = GDC(
        self_loop_weight=1.0,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs={
            'method': 'ppr',
            'alpha': 1.0 - alpha,
            'eps': 1e-4,
        },
        sparsification_kwargs={
            'method': 'topk',
            'k': topk_final,
            'dim': 1,
        },
        exact=True
    )

    personalization = torch.zeros(graph.num_nodes, 1, device=device)
    personalization[seed_node_indices] = 1.0 / len(seed_node_indices)

    tmp_data = Data(x=personalization, edge_index=graph.edge_index)
    out = gdc(tmp_data)
    ppr_scores = out.x.view(-1)

    # 获取最终节点集合
    selected_nodes = ppr_scores.to_dense().nonzero(as_tuple=False).view(-1)
    selected_nodes = torch.unique(torch.cat([selected_nodes, seed_node_indices]))
    selected_node_set = set(selected_nodes.tolist())

    # === 步骤3：构造增强连接边集合（只要一端在节点集合中）===
    final_edge_indices = []
    for i in range(graph.edge_index.shape[1]):
        u = graph.edge_index[0, i].item()
        v = graph.edge_index[1, i].item()
        # 强化连边策略（只要u或v在选中节点中）
        if u in selected_node_set or v in selected_node_set:
            final_edge_indices.append(i)

    # === 步骤4：导出节点和边的文本描述 ===
    nodes_df = textual_nodes.iloc[selected_nodes.cpu().numpy()]
    edges_df = textual_edges.iloc[final_edge_indices]
    if edges_df.empty:
        return []
        
    # 1. 创建一个从 node_id 到 node_attr 的快速查找映射（字典）
    node_map = pd.Series(nodes_df.node_attr.values, index=nodes_df.node_id).to_dict()

    triplets = []
    # 2. 遍历每一条边
    for _, edge in edges_df.iterrows():
        # 3. 使用映射查找源节点和目标节点的文本属性
        src_attr = node_map.get(edge['src'])
        dst_attr = node_map.get(edge['dst'])
        edge_attr = edge['edge_attr']

        # 确保所有部分都找到了
        if src_attr is not None and dst_attr is not None:
            triplets.append((src_attr, edge_attr, dst_attr))
    
    return triplets










def retrieval_pcst_attention_heads_desc_only_optimized(graph, q_emb_heads, head_importance_scores, textual_nodes, textual_edges, 
                                                       topk=3, topk_e=3, cost_e=0.5):
    """
    (优化版) 使用多头注意力向量和动态重要性分数来计算PCST的奖赏值，并仅返回子图的文本描述。

    参数 (Inputs):
        graph (Data): PyTorch Geometric的图数据对象，必须包含以下属性：
                      - .x_attention_heads: 节点的“多头”特征张量，形状为 [N, n_heads, D]。
                      - .e_attention_heads: 边的“多头”特征张量，形状为 [E, n_heads, D]。
                      - .edge_index: 图的连接信息，形状为 [2, E]。
                      - .num_nodes: 节点总数。
        q_emb_heads (torch.Tensor): 查询的“多头”嵌入向量，形状为 [1, n_heads, D]。
        head_importance_scores (torch.Tensor): 包含'n_heads'个浮点数的张量，代表每个头在当前查询下的重要性分数。
        textual_nodes (pd.DataFrame): 节点的文本描述信息。
        textual_edges (pd.DataFrame): 边的文本描述信息。
        topk (int, optional): 选取top-k个最相关的节点赋予奖赏，默认为3。
        topk_e (int, optional): 选取top-k个最相关的边赋予奖赏，默认为3。
        cost_e (float, optional): 边的基础成本，默认为0.5。

    返回 (Output):
        str: 一个CSV格式的字符串，包含了最终被选中的子图的所有节点和边的文本描述。
    """
    print("--- 正在运行: 基于多头注意力和PCST的检索算法 (优化版) ---")
    c = 0.01  # 用于奖赏值计算的一个小的衰减系数
    # 边界条件处理：如果图的文本描述为空，则直接返回，不进行计算。
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        return textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # === 步骤一：奖赏值计算 (Prize Calculation) ===
    # 将权重张量的形状调整为 (1, n_heads) 以便进行广播乘法。
    head_weights = head_importance_scores.view(1, -1)
    # --- 计算节点奖赏 n_prizes ---
    if topk > 0:
        # 1. 计算查询的每个头与节点对应的每个头的余弦相似度。
        graph.x_attention_heads = torch.tensor(graph.x_attention_heads, dtype=q_emb_heads.dtype, device=q_emb_heads.device)

        node_sim_scores = F.cosine_similarity(q_emb_heads, graph.x_attention_heads, dim=-1) # 输出形状: [N, n_heads]
        # 2. 根据每个头的重要性权重，对相似度分数进行加权求和，得到每个节点的总分。
        initial_node_prizes = torch.sum(node_sim_scores * head_weights, dim=1) # 输出形状: [N]
        # 3. 找到分数最高的 topk 个节点。
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(initial_node_prizes, topk, largest=True)
        # 4. 创建最终的节点奖赏张量，只为top-k节点赋予从k到1的递减奖赏。
        n_prizes = torch.zeros_like(initial_node_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes) # 如果topk=0, 则所有节点奖赏为0。

    # --- 计算边奖赏 e_prizes ---
    if topk_e > 0:
        # 1. 计算查询的每个头与边对应的每个头的余弦相似度。
        graph.e_attention_heads = torch.tensor(graph.e_attention_heads, dtype=q_emb_heads.dtype, device=q_emb_heads.device)

        edge_sim_scores = F.cosine_similarity(q_emb_heads, graph.e_attention_heads, dim=-1) # 输出形状: [E, n_heads]
        # 2. 加权求和得到每条边的初始奖赏值。
        e_prizes = torch.sum(edge_sim_scores * head_weights, dim=1) # 输出形状: [E]
        # 3. 对边的奖赏值进行更精细的重新加权和排序，以拉开差距。
        topk_e = min(topk_e, e_prizes.unique().size(0))
        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0 # 低于门槛的奖赏清零
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            if indices.sum() > 0: # 确保分母不为0
                value = min((topk_e-k)/indices.sum(), last_topk_e_value)
                e_prizes[indices] = value
                last_topk_e_value = value*(1-c)
        # 4. 动态调整边的基础成本，确保最高的边奖赏值不会比成本高太多，保证算法能运行。
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges) # 如果topk_e=0, 则所有边奖赏为0。
    
    # === 步骤二：PCST问题转化与求解 (矢量化优化版) ===
    # --- 数据准备 ---
    # 将所有需要在CPU上用Numpy处理的张量一次性转移，避免循环中频繁的设备切换。
    e_prizes_cpu = e_prizes.cpu()
    edge_index_cpu = graph.edge_index.cpu()
    n_prizes_cpu = n_prizes.cpu().numpy()

    # 1. 使用布尔掩码对所有边进行分类，分为“常规边”和“虚拟边”。
    is_virtual_mask = e_prizes_cpu > cost_e  # 奖赏大于成本的边，需要转化为虚拟节点。
    is_regular_mask = ~is_virtual_mask   # 奖赏小于等于成本的边。

    # 2. 处理常规边 (prize <= cost)
    regular_edges_indices = torch.where(is_regular_mask)[0] # 获取常规边的原始索引
    regular_edges = edge_index_cpu[:, regular_edges_indices].T.numpy() # 获取常规边的端点
    regular_costs = (cost_e - e_prizes_cpu[is_regular_mask]).numpy() # 计算常规边的成本
    
    # 3. 处理虚拟边 (prize > cost)
    virtual_edges_indices = torch.where(is_virtual_mask)[0] # 获取虚拟边的原始索引
    num_virtual = len(virtual_edges_indices) # 虚拟边的数量
    # 为每个虚拟边创建一个新的、唯一的虚拟节点ID。
    virtual_node_ids = np.arange(graph.num_nodes, graph.num_nodes + num_virtual)
    
    # 获取虚拟边的源节点和目标节点
    src_virtual = edge_index_cpu[0, is_virtual_mask].numpy()
    dst_virtual = edge_index_cpu[1, is_virtual_mask].numpy()
    
    # 将原始边 (u,v) 拆分为 (u, v_node) 和 (v_node, v)
    virtual_edges_part1 = np.stack([src_virtual, virtual_node_ids], axis=1)
    virtual_edges_part2 = np.stack([virtual_node_ids, dst_virtual], axis=1)
    virtual_edges = np.concatenate([virtual_edges_part1, virtual_edges_part2], axis=0)
    virtual_costs = np.zeros(2 * num_virtual) # 虚拟边的成本为0
    # 虚拟节点的奖赏等于原边的“超额奖赏” (prize - cost)
    virtual_n_prizes = (e_prizes_cpu[is_virtual_mask] - cost_e).numpy()

    # 4. 组合所有输入，形成PCST求解器的最终输入。
    final_edges = np.concatenate([regular_edges, virtual_edges], axis=0)
    final_costs = np.concatenate([regular_costs, virtual_costs], axis=0)
    final_prizes = np.concatenate([n_prizes_cpu, virtual_n_prizes], axis=0)
    
    # 创建映射字典，用于在求解后将结果映射回原始的边索引。
    mapping_e = {i: idx.item() for i, idx in enumerate(regular_edges_indices)}
    mapping_n = {v_id: idx.item() for v_id, idx in zip(virtual_node_ids, virtual_edges_indices)}
    
    # 5. 设置PCST参数并调用求解器。
    root, num_clusters, pruning, verbosity_level = -1, 1, 'gw', 0
    # `pcst_fast`返回被选中的顶点和边的索引（在`final_edges`输入中的索引）。
    vertices, pcst_edges_result = pcst_fast(final_edges.tolist(), final_prizes, final_costs, root, num_clusters, pruning, verbosity_level)

    # === 步骤三：结果解析与描述生成 ===
    num_regular_edges = len(regular_edges) # 常规边的数量，用于区分返回的边索引
    # 从返回的顶点中分离出真实节点（ID < num_nodes）
    selected_nodes = vertices[vertices < graph.num_nodes]
    # 使用映射，将PCST返回的常规边索引转换回原始图的边索引。
    selected_edges = [mapping_e[e_idx] for e_idx in pcst_edges_result if e_idx < num_regular_edges]
    # 从返回的顶点中分离出虚拟节点
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0: # 如果有虚拟节点被选中
        # 使用映射，将选中的虚拟节点ID转换回它们所代表的原始高奖赏边的索引。
        virtual_edges_indices_from_result = [mapping_n[v_idx] for v_idx in virtual_vertices]
        selected_edges.extend(virtual_edges_indices_from_result) # 将这些边也加入最终列表

    # 确保所有被选中边的端点都包含在最终的节点集中，保证描述的子图是连贯的。
    if len(selected_edges) > 0:
        edge_index_for_nodes = graph.edge_index[:, selected_edges]
        final_selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index_for_nodes[0].cpu().numpy(), edge_index_for_nodes[1].cpu().numpy()]))
    else:
        final_selected_nodes = selected_nodes
        
    # 根据最终确定的节点和边索引，从DataFrame中提取文本描述。
    nodes_df = textual_nodes.iloc[final_selected_nodes]
    edges_df = textual_edges.iloc[selected_edges]
    # 将描述转换为CSV格式的字符串。
    node_map = pd.Series(nodes_df.node_attr.values, index=nodes_df.node_id).to_dict()

    triplets = []
    # 2. 遍历每一条边
    for _, edge in edges_df.iterrows():
        # 3. 使用映射查找源节点和目标节点的文本属性
        src_attr = node_map.get(edge['src'])
        dst_attr = node_map.get(edge['dst'])
        edge_attr = edge['edge_attr']

        # 确保所有部分都找到了，避免因数据不一致而出错
        if src_attr is not None and dst_attr is not None:
            triplets.append((src_attr, edge_attr, dst_attr))
    
    return triplets


import json

def load_seeds(path: str) -> dict[str, dict[str, list[int]]]:

    seeds: dict[str, dict[str, list[int]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 用 query_index 作为 key
            idx = str(obj.get('query_index', obj.get('id')))
            results = obj.get('results', {})
            # 从 multirag_* 字段取值，默认空列表
            entities = results.get('multirag_entity', [])
            relations = results.get('multirag_relation', [])
            # 确保是 list[int]
            if not isinstance(entities, list):
                entities = [entities]
            if not isinstance(relations, list):
                relations = [relations]
            seeds[idx] = {
                'entity': [int(x) for x in entities],
                'relation': [int(x) for x in relations],
            }
    return seeds



def load_standard(path: str) -> dict[str, dict[str, list[int]]]:

    seeds: dict[str, dict[str, list[int]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 用 query_index 作为 key
            idx = str(obj.get('query_index', obj.get('id')))
            results = obj.get('results', {})
            # 从 multirag_* 字段取值，默认空列表
            entities = results.get('standard-rag_entity', [])
            relations = results.get('standard-rag_relation', [])
            # 确保是 list[int]
            if not isinstance(entities, list):
                entities = [entities]
            if not isinstance(relations, list):
                relations = [relations]
            seeds[idx] = {
                'entity': [int(x) for x in entities],
                'relation': [int(x) for x in relations],
            }
    return seeds

def convert_indices_to_attrs(node_indices: list[int], edge_indices: list[int], nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    一个辅助函数，用于将节点和边的索引列表转换为可读的文本属性列表。

    Args:
        node_indices (List[int]): 节点的ID索引列表。
        edge_indices (List[int]): 边的行号索引列表。
        nodes_df (pd.DataFrame): 包含节点信息的DataFrame。
        edges_df (pd.DataFrame): 包含边信息的DataFrame。

    Returns:
        Tuple[List[str], List[str]]: 一个元组，包含(节点属性列表, 边属性列表)。
    """
    # 为了快速查找，将 nodes_df 的索引设置为 node_id 列
    nodes_df_indexed = nodes_df.set_index('node_id')
    
    # 转换节点索引 -> 节点属性
    entity_attrs = []
    for node_id in node_indices:
        try:
            attr = nodes_df_indexed.loc[node_id]['node_attr']
            entity_attrs.append(attr)
        except KeyError:
            print(f"Warning: node_id {node_id} not found in nodes CSV. Skipping.")

    # 转换边索引 -> 边属性
    edge_attrs = []
    for edge_index in edge_indices:
        try:
            attr = edges_df.iloc[edge_index]['edge_attr']
            edge_attrs.append(attr)
        except IndexError:
            print(f"Warning: edge_index {edge_index} is out of bounds for edges CSV. Skipping.")
            
    return entity_attrs, edge_attrs

def main():
    p = argparse.ArgumentParser(
        description="Batch run union/PPR/PCST retrieval for all subgraphs under a base path"
    )
    p.add_argument(
        '-b','--base-path', type=str, required=True,
        help='Root folder containing nodes/, edges/, graphs/, q_attention_embs.pt, head_scores.pt, seeds.csv'
    )
    p.add_argument(
        '-o','--output', type=str, default='GH_retrieve.jsonl',
        help='JSONL file to write the batch results'
    )
    args = p.parse_args()

    base = args.base_path
    # 1. 读 seeds 和 standards
    seeds = load_seeds(os.path.join(base, 'mha_retrieve.jsonl'))
    standards = load_standard(os.path.join(base, 'mha_retrieve.jsonl'))
    
    # 2. 预载 attention head scores 和 q_emb_heads
    q_emb_all = torch.load(os.path.join(base, 'q_attention_embs.pt'))
    head_scores = torch.load(os.path.join(base, 'head_scores.pt'))

    out_f = open(args.output, 'w', encoding='utf-8')
    problematic_ids = {'1522', '1529', '1340'}


    for idx, seed in seeds.items():
        if idx in problematic_ids:
            # 如果是，则构造一个空的记录
            empty_record = {
                "id": idx,
                "seed_graph": [],
                "standard_graph": [],
                "union": [],
                "ppr": [],
                "pcst": [],
            }
            # 将空记录写入文件
            out_f.write(json.dumps(empty_record, ensure_ascii=False) + "\n")
            print(f"[{idx}] is a known problematic ID. Skipped, wrote empty record.")
            # 使用 continue 跳过本次循环的剩余部分，直接进入下一次循环
            continue
        # 3a. 加载图和文本描述
        graph: Data = torch.load(os.path.join(base, 'graphs', f'{idx}.pt'), weights_only=False)
        nodes_df = pd.read_csv(os.path.join(base, 'nodes', f'{idx}.csv'))
        edges_df = pd.read_csv(os.path.join(base, 'edges', f'{idx}.csv'))

        # <<< MODIFIED: 使用辅助函数处理 seed 和 standard >>>
        # --- 处理 seed ---
        seed_nodes, seed_edges = seed['entity'], seed['relation']
        seed_entity_attrs, seed_edge_attrs = convert_indices_to_attrs(
            seed_nodes, seed_edges, nodes_df, edges_df
        )

        # --- 处理 standard ---
        standard = standards.get(idx)
        if standard:
            standard_nodes, standard_edges = standard['entity'], standard['relation']
            standard_entity_attrs, standard_edge_attrs = convert_indices_to_attrs(
                standard_nodes, standard_edges, nodes_df, edges_df
            )
        else:
            # 如果在standards中找不到对应的id，则设置为空列表
            standard_entity_attrs, standard_edge_attrs = [], []
        # >>> END MODIFICATION

        # --- retrieval 部分保持不变，仍然基于 seed ---
        # 4. Union
        union_desc = build_desc_from_indices_standard(
            graph=graph, topk_n_indices=seed_nodes, topk_e_indices=seed_edges,
            textual_nodes=nodes_df, textual_edges=edges_df
        )

        # 5. PPR
        ppr_desc = prr_from_topk(
            graph=graph, topk_n_indices=seed_nodes, topk_e_indices=seed_edges,
            textual_nodes=nodes_df, textual_edges=edges_df,
            topk_final=5, alpha=0.15
        )

        # 6. PCST
        q_emb = q_emb_all[int(idx)]
        if not isinstance(q_emb, torch.Tensor):
            q_emb = torch.tensor(q_emb)
        pcst_desc = retrieval_pcst_attention_heads_desc_only_optimized(
            graph=graph, q_emb_heads=q_emb, head_importance_scores=head_scores,
            textual_nodes=nodes_df, textual_edges=edges_df,
            topk=3, topk_e=3, cost_e=0.5
        )

        # 7. 写一行 JSONL
        # <<< MODIFIED: 更新输出记录 >>>
        record = {
            "id": idx,
            "seed_graph": seed_entity_attrs + seed_edge_attrs,
            "standard_graph": standard_entity_attrs + standard_edge_attrs, # 新增 standard 的可读图
            "union": union_desc,
            "ppr": ppr_desc,
            "pcst": pcst_desc
        }
        # >>> END MODIFICATION
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx}] done")

    out_f.close()
    print("All finished. Results in", args.output)

if __name__ == '__main__':
    # 假设您的项目结构，您可能需要调整 'your_module'
    # from some.path import your_module
    main()

