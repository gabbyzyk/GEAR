
import psycopg2
import numpy as np
from tqdm import tqdm
from pgvector.psycopg2 import register_vector
import cupy as cp

from typing import Any, Union
from enum import Enum

from multirag.dataset import (
    Article
)
from multirag.embed import (
    Embedding,
    FullEmbeddings,
    ArticleEmbeddings,
)


class DistanceMetric(Enum):
    """
    Class to enumerate the distance metrics.
    """
    COSINE = '<=>'
    DOT = '<#>'
    MANHATTAN = '<+>'
    EUCLIDEAN = '<->'

    def __str__(self) -> str:
        return self.name.lower()


class VectorDB:
    """
    Class for a vector database.
    """

    def __init__(
            self,
            distance_metric: DistanceMetric,
            name: str = "vector-database",
            host: str = "localhost",
            port: int = 5430,
            user: str = "postgres",
            password: str = "password",
    ) -> None:
        """
        Initialize the VectorDB instance with the distance metric and the database parameters.

        :param distance_metric: Distance metric to use.
        :type distance_metric: DistanceMetric
        :param name: Name of the vector database instance. Defaults to vector-database.
        :type name: str
        :param host: Host of the vector database instance. Defaults to localhost.
        :type host: str
        :param port: Port of the vector database instance. Defaults to 5431.
        :type port: int
        :param user: User name for the vector database instance. Defaults to postgres.
        :type user; str
        :param password: Password for the vector database instance. Defaults to password.
        :type password: str
        """
        self.dist_metric = distance_metric
        self._conn = psycopg2.connect(
            database=name,
            host=host,
            port=port,
            user=user,
            password=password
        )
        self._similarity_cache: dict[Any, list] = {}
        self._initialize()

    def _initialize(self) -> None:
        """
        Register vector type with the database-connection.
        """
        register_vector(self._conn)

    def _fetch(self, sql: str, args: Union[tuple, dict] = ()) -> list[Any]:
        """
        Fetch data from an SQL query.

        :param sql: SQL query to execute.
        :type sql: str
        :param args: Arguments for the SQL query. Defaults to empty.
        :type args: Union[tuple, dict]
        :return: Results of the executed query as a list of elements.
        :rtype: list[Any]
        """
        curs = self._conn.cursor()
        curs.execute(sql, args)
        res = curs.fetchall()
        curs.close()
        return res

    def _execute(self, sql: str, args: Union[tuple, dict] = ()) -> None:
        """
        Execute an SQL query.

        :param sql: SQL query to execute.
        :type sql: str
        :param args: Arguments for the SQL query. Defaults to empty.
        :type args: Union[tuple, dict]
        """
        curs = self._conn.cursor()
        curs.execute(sql, args)
        curs.close()
        return

    @property
    def empty(self) -> bool:
        """
        Check if the database is empty.

        :return: True if the database is empty. Otherwise False.
        :rtype: bool
        """
        return len(self._fetch("SELECT * FROM articles LIMIT 1;")) == 0

    @property
    def attention_scales(self) -> list[float]:
        """
        Get the attention scales from the respective database table.
        This and the cut standard scales should be computed during the first
        initialisation of the vector database.

        :return: Attention scales.
        :rtype: list[float]
        """
        return self._fetch("SELECT scales FROM attention_scales LIMIT 1;")[0][0].tolist()

    @property
    def cut_standard_scales(self) -> list[float]:
        """
        Get the cut standard scales from the respective database table.

        :return: Cut standard scales:
        :rtype: list[float]
        """
        return self._fetch("SELECT scales FROM cut_standard_scales LIMIT 1;")[0][0].tolist()

    # def _compute_attention_scales(self, layer_idx: int) -> np.ndarray:
    #     """
    #     Compute the scales for the attention heads.

    #     The scale for a head is the product of the mean norm of that head's embeddings,
    #     and the mean pairwise cosine distance between embeddings in the head's embedding
    #     space.

    #     The computation takes some time. This happens only once, so you should not manually
    #     call this method, but instead call the `get_attention_scales` method.

    #     :param layer_idx: Layer to target for the computation.
    #     :type layer_idx: int
    #     :return: Attention head scales.
    #     :rtype: np.ndarray
    #     """
    #     all_heads = ", ".join(f"head{head:02}" for head in range(32))
    #     head_embs = self._fetch(f"SELECT {all_heads} FROM attention WHERE layer_index = %s;", (layer_idx,))

    #     all_heads_avg = ", ".join(f"AVG(head{head:02} {DistanceMetric.COSINE.value} %s) " for head in range(32))
    #     dist_sql = f"SELECT {all_heads_avg} FROM attention WHERE layer_index = %s;"
    #     pairwise_distances = np.zeros((len(head_embs), 32))

    #     description = f"Calculating pairwise attentions distances at layer {layer_idx}"
    #     for i, emb in enumerate(tqdm(head_embs, description)):
    #         pairwise_distances[i] = self._fetch(dist_sql, (*emb, layer_idx))[0]

    #     all_heads_avg = ", ".join(f"AVG(head{head:02} {DistanceMetric.EUCLIDEAN.value} %(zero)s) " for head in range(32))
    #     dist_sql = f"SELECT {all_heads_avg} FROM attention WHERE layer_index = %(layer)s;"
    #     norms = np.array(self._fetch(dist_sql, dict(zero=np.zeros(128), layer=layer_idx))[0])

    #     return norms * pairwise_distances.mean(axis=0)
    
    # def _compute_attention_scales(self, layer_idx: int) -> np.ndarray:
    #     """
    #     用 CuPy 在 GPU 上并行计算 attention head scales。
    #     """
    #     # 1) 拿出所有 head 向量 (N,32,D)
    #     head_rows = self._fetch(
    #         "SELECT " + ", ".join(f"head{h:02}" for h in range(32)) +
    #         " FROM attention WHERE layer_index = %s;",
    #         (layer_idx,)
    #     )
    #     # 转成 NumPy，再到 CuPy
    #     arr = cp.array(head_rows, dtype=cp.float32)   # shape (N, 32, D)

    #     # 2) 计算平均范数 norms[i] = mean_j ||v_{j,i}||₂
    #     norms = cp.linalg.norm(arr, axis=2).mean(axis=0)  # (32,)

    #     # 3) 归一化后计算平均余弦距离
    #     emb_norm = arr / norms[None,:,None]               # 广播，shape (N,32,D)
    #     # 对每个 head 做 N×N 矩阵乘再求均值
    #     mean_cos = cp.empty(32, dtype=cp.float32)
    #     for i in range(32):
    #         x = emb_norm[:, i, :]                         # (N,D)
    #         # x @ x.T：每对向量余弦相似度，求平均
    #         mean_cos[i] = (x @ x.T).mean()
    #     mean_dist = 1 - mean_cos                          # 余弦距离

    #     # 4) scale = norms * mean_dist，拉回到 CPU
    #     scales = (norms * mean_dist).get()                # np.ndarray (32,)
    #     return scales

    def _compute_attention_scales(self, layer_idx: int) -> np.ndarray:
        """
        用 CuPy 在 GPU 上并行计算 attention head scales (优化版)。
        """
        # 1) 拿出所有 head 向量 (N,32,D)
        head_rows = self._fetch(
            "SELECT " + ", ".join(f"head{h:02}" for h in range(32)) +
            " FROM attention WHERE layer_index = %s;",
            (layer_idx,)
        )
        arr = cp.array(head_rows, dtype=cp.float32)   # shape (N, 32, D)

        # 2) 计算平均范数 norms[i] = mean_j ||v_{j,i}||₂
        norms = cp.linalg.norm(arr, axis=2).mean(axis=0)  # (32,)

        # 3) 归一化后计算平均余弦距离
        emb_norm = arr / norms[None, :, None]               # 广播，shape (N,32,D)
        
        mean_cos = cp.empty(32, dtype=cp.float32)
        N = emb_norm.shape[0] # 获取节点总数 N

        # <<< MODIFIED: 使用数学优化方法替代昂贵的矩阵乘法 >>>
        for i in range(32):
            x = emb_norm[:, i, :]                         # (N,D)
            
            # 计算所有向量的和
            col_sum = cp.sum(x, axis=0)                   # shape (D,)
            
            # 计算总和的点积，等价于 (x @ x.T) 中所有元素的和
            total_sum_of_cosines = cp.dot(col_sum, col_sum)
            
            # 计算平均余弦相似度
            if N > 0:
                mean_cos[i] = total_sum_of_cosines / (N * N)
            else:
                mean_cos[i] = 0
        # >>> END MODIFICATION
                
        mean_dist = 1 - mean_cos                          # 余弦距离

        # 4) scale = norms * mean_dist，拉回到 CPU
        scales = (norms * mean_dist).get()                # np.ndarray (32,)
        return scales

    def _update_attention_scales(self) -> None:
        """
        Update the scales for the attention heads for all layers that are in the database.

        This computation takes even more time.
        """
        rows = self._fetch("SELECT DISTINCT layer_index FROM attention ORDER BY layer_index;")
        sql = "INSERT INTO attention_scales (scales, layer_index) VALUES (%s, %s);"
        for layer_idx in (row[0] for row in rows):
            attention_scales = self._compute_attention_scales(layer_idx)
            self._execute(sql, (attention_scales, layer_idx))

    def _compute_cut_standard_scales(self) -> np.ndarray:
        """
        Compute the scales of the segmented standard embedding.

        :return: Scales of the segmented standard embedding.
        :rtype: np.ndarray
        """
        all_segments = ", ".join(f"segment{seg:02}" for seg in range(32))
        seg_embs = self._fetch(f"SELECT {all_segments} FROM cut_standard;")

        all_segs_avg = ", ".join(f"AVG(segment{seg:02} {DistanceMetric.COSINE.value} %s)" for seg in range(32))
        dist_sql = f"SELECT {all_segs_avg} FROM cut_standard;"

        pairwise_distances = np.zeros((len(seg_embs), 32))
        for i, emb in enumerate(tqdm(seg_embs, "Calculating pairwise segment distances")):
            pairwise_distances[i] = self._fetch(dist_sql, emb)[0]

        all_segs_avg = ", ".join(f"AVG(segment{seg:02} {DistanceMetric.EUCLIDEAN.value} %(zero)s)" for seg in range(32))
        dist_sql = f"SELECT {all_segs_avg} FROM cut_standard;"
        norms = np.array(self._fetch(dist_sql, dict(zero=np.zeros(128)))[0])

        return norms * pairwise_distances.mean(axis=0)

    def _update_cut_standard_scales(self) -> None:
        """
        Update the scales of the segmented standard embedding in the database.
        """
        sql = "INSERT INTO cut_standard_scales (scales) VALUES (%s);"
        cut_scales = self._compute_cut_standard_scales()
        self._execute(sql, (cut_scales,))

    def _add_standard_embedding(self, article_id: int, embedding: Embedding) -> None:
        """
        Add the standard embedding of an article into a database table.

        :param article_id: Article ID to embed.
        :type article_id: int
        :param embedding: Embedding of the article.
        :type embedding: Embedding
        """
        sql = "INSERT INTO standard (article_id, embedding) VALUES (%s, %s);"
        self._execute(sql, (article_id, embedding))

    @staticmethod
    def _split_embedding(embedding: Embedding) -> list[Embedding]:
        """
        Split the embedding into 32 chunks.

        :param embedding: Embedding to be split.
        :type embedding: Embedding
        :return: List of embedding chunks.
        :rtype: list[Embedding]
        """
        chunk_size, rem = divmod(len(embedding), 32)
        assert rem == 0, "embedding length must be divisible by 32"
        chunks = []
        for i in range(0, len(embedding), chunk_size):
            chunks.append(embedding[i:(i + chunk_size)])
        return chunks

    def _add_cut_embeddings(self, article_id: int, embedding: Embedding) -> None:
        """
        Add the cut standard embedding of an article into a database table.

        :param article_id: Article ID to embed.
        :type article_id: int
        :param embedding: Embedding of the article.
        :type embedding: Embedding
        """
        all_segments = ", ".join(f"segment{seg:02}" for seg in range(32))
        all_values = ", ".join("%s" for _ in range(33))
        sql = f"INSERT INTO cut_standard (article_id, {all_segments}) VALUES ({all_values});"
        chunks = self._split_embedding(embedding)
        self._execute(sql, (article_id, *chunks))

    def _add_attention_embeddings(self, article_id: int, layer_idx: int, embeddings: list[Embedding]) -> None:
        """
        Add the attention embeddings of an article for a specific layer into a database table.

        :param article_id: Article ID to embed.
        :type article_id: int
        :param layer_idx: Layer of the attention embedding.
        :type layer_idx: int
        :param embeddings: Attention embeddings of the article in layer layer_idx.
        :type embeddings: list[Embedding]
        """
        all_heads = ", ".join(f"head{head:02}" for head in range(32))
        all_values = ", ".join("%s" for _ in range(34))
        sql = f"INSERT INTO attention (article_id, layer_index, {all_heads}) VALUES ({all_values});"
        self._execute(sql, (article_id, layer_idx, *embeddings))

    def _add_article(self, article_emb: ArticleEmbeddings) -> None:
        """
        Add an article and their embeddings (standard, cut standard, attention) into the database.

        :param article_emb: Article with attributes and embeddings.
        :type article_emb: ArticleEmbeddings
        """
        article = article_emb.article
        sql = "INSERT INTO articles (title, content, label) VALUES (%s, %s, %s) RETURNING id;"
        article_id: int = self._fetch(sql, (article.title, article.text, article.label))[0][0]

        embeddings = article_emb.embeddings
        self._add_standard_embedding(article_id, embeddings.standard_embedding)
        # self._add_cut_embeddings(article_id, embeddings.standard_embedding)
        for layer_idx, layer_embeddings in embeddings.layer_embeddings.items():
            self._add_attention_embeddings(article_id, layer_idx, layer_embeddings.attention_heads)

    def add_articles(self, articles: list[ArticleEmbeddings]) -> None:
        """
        Add all the provided articles to the database, then compute their attention-
        and standard-scales and store the results as well.

        The database contains 5 tables:
        - "standard" for the standard embeddings
        - "attention" for the attention embeddings
        - "cut_standard" for the cut standard embeddings
        - "attention_scales" and "cut_standard_scales" for the scales for the
          respective embeddings' components.

        The database in its current form only computes these scales after adding the
        articles, which only happens during the initial startup. Afterwards, it only
        loads the scales from the respective tables. It is therefore not designed to
        support changes within the dataset or articles (adding / removing / changing
        embeddings). Note that when computing and storing the attention / standard
        scales, the database also does not ensure the according tables are empty. If
        this function is called multiple times, with different batches of articles, it
        is therefore not guaranteed that when getting the attention scales, the correct
        scales for the dataset's current form are returned.

        :param articles: Articles and their embeddings to add to the database.
        :type articles: list[ArticleEmbeddings]
        """
        for article_emb in tqdm(articles, "Importing article embeddings"):
            self._add_article(article_emb)

        self._conn.commit()
        self._update_attention_scales()
        # self._update_cut_standard_scales()
        self._conn.commit()

    def clear(self) -> None:
        """
        Delete all records from all tables in the database.
        """
        self._execute("DELETE FROM cut_standard_scales;")
        self._execute("DELETE FROM attention_scales;")
        self._execute("DELETE FROM cut_standard;")
        self._execute("DELETE FROM attention;")
        self._execute("DELETE FROM standard;")
        self._execute("DELETE FROM articles;")
        self._conn.commit()

    def standard_search(self, emb: FullEmbeddings, n: int = 32) -> list[tuple[float, Article]]:
        """
        Do a similarity search for the provided query within the standard embedding space.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param n: Number of similar embeddings to pick. Defaults to 32.
        :type n: int
        :returns: List of tuples (distance, Article) with top :n: selections.
        :rtype: list[tuple[float, Article]]
        """
        embedding = np.array(emb.standard_embedding)
        cache_key = ('standard', embedding.data.tobytes())
        res: list[tuple[float, int]] = self._similarity_cache.get(cache_key, [])
        if len(res) < n:
            sql = f"""
                SELECT embedding {self.dist_metric.value} %s 
                AS distance, articles.title, articles.content, articles.label
                FROM standard 
                INNER JOIN articles ON standard.article_id = articles.id
                ORDER BY embedding {self.dist_metric.value} %s 
                LIMIT %s;
            """
            res = self._fetch(sql, (embedding, embedding, n))
            self._similarity_cache[cache_key] = res

        articles: list[tuple[float, Article]] = []
        for distance, title, content, label in res[:n]:
            articles.append((distance, Article(title, content, label)))

        return articles

    def attention_search(
            self,
            emb: FullEmbeddings,
            layer_idx: int,
            n: int = 32
    ) -> list[list[tuple[float, Article]]]:
        """
        Do a similarity search for the provided query within the attention embedding spaces.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param layer_idx: Layer to target.
        :type layer_idx: int
        :param n: Number of similar embeddings to pick for each attention head. Defaults to 32.
        :type n: int
        :returns: List of 32 lists of tuples (distance, Article) with top :n: selections for each head.
        :rtype: list[list[tuple[float, Article]]]
        """
        res: list[list[tuple[float, Article]]] = []
        embeddings = np.array(emb.layer_embeddings[layer_idx].attention_heads)

        for i, embedding in enumerate(embeddings):
            cache_key = (f'attention-{layer_idx}-{i}', embedding.data.tobytes())
            head_res = self._similarity_cache.get(cache_key, [])
            if len(head_res) < n:
                sql = f"""
                    SELECT head{i:02} {self.dist_metric.value} %s 
                    AS distance, articles.title, articles.content, articles.label
                    FROM attention
                    INNER JOIN articles ON attention.article_id = articles.id
                    WHERE layer_index = %s 
                    ORDER BY head{i:02} {self.dist_metric.value} %s 
                    LIMIT %s;
                """
                head_res = self._fetch(sql, (embedding, layer_idx, embedding, n))
                self._similarity_cache[cache_key] = head_res

            articles: list[tuple[float, Article]] = []
            for distance, title, content, label in head_res[:n]:
                articles.append((distance, Article(title, content, label)))

            res.append(articles)

        return res
    

    def cut_standard_search(self, emb: FullEmbeddings, n: int = 32) -> list[list[tuple[float, Article]]]:
        """
        Do a similarity search for the provided query within the cut standard embedding spaces.

        :param emb: Query embedding.
        :type emb: FullEmbeddings
        :param n: Number of similar embedding to pick within each cut standard embedding segment. Defaults to 32.
        :type n: int
        :returns: List of 32 lists of tuples (distance, title) with top :n: selections for each segment.
        :rtype: list[list[tuple[float, Article]]]
        """
        res: list[list[tuple[float, Article]]] = []
        embedding = emb.standard_embedding
        chunks = np.array(self._split_embedding(embedding))

        for i, embedding in enumerate(chunks):
            cache_key = (f'cut_standard-{i}', embedding.data.tobytes())
            seg_res = self._similarity_cache.get(cache_key, [])
            if len(seg_res) < n:
                sql = f"""
                    SELECT segment{i:02} {self.dist_metric.value} %s 
                    AS distance, articles.title, articles.content, articles.label
                    FROM cut_standard 
                    INNER JOIN articles ON cut_standard.article_id = articles.id
                    ORDER BY segment{i:02} {self.dist_metric.value} %s 
                    LIMIT %s;
                """
                seg_res = self._fetch(sql, (embedding,  embedding, n))
                self._similarity_cache[cache_key] = seg_res

            articles: list[tuple[float, Article]] = []
            for distance, title, content, label in seg_res[:n]:
                articles.append((distance, Article(title, content, label)))

            res.append(articles)

        return res


    def get_attention_scales_for_layer(self, layer_idx: int) -> list[float]:
        """
        从数据库中获取指定层的注意力头缩放因子 (scales)。

        :param layer_idx: 要查询的目标层索引。
        :type layer_idx: int
        :return: 一个包含32个浮点数的列表，代表该层每个头的重要性分数。
        :rtype: list[float]
        """
        sql = "SELECT scales FROM attention_scales WHERE layer_index = %s LIMIT 1;"
        result = self._fetch(sql, (layer_idx,))
        if not result:
            raise ValueError(f"数据库中未找到第 {layer_idx} 层的 attention scales。")
        return result[0][0].tolist()

    def entity_search(
        self,
        query_id: int,
        emb: FullEmbeddings,
        n: int = 32
    ) -> list[tuple[float, Article]]:
        """
        只在 graph_id=query_id 对应的实体子集里做标准向量相似度检索。
        """
        vec = np.array(emb.standard_embedding)
        cache_key = (f"entity-{query_id}", vec.data.tobytes())
        results = self._similarity_cache.get(cache_key)
        if not results or len(results) < n:
            sql = f"""
                WITH gids AS (
                SELECT head_gid AS gid FROM graph_triples
                    WHERE graph_id = (
                    SELECT id FROM graphs WHERE question_id = %s
                    )
                UNION
                SELECT tail_gid FROM graph_triples
                    WHERE graph_id = (
                    SELECT id FROM graphs WHERE question_id = %s
                    )
                )
                SELECT
                s.embedding {self.dist_metric.value} %s AS distance,
                a.title, a.content, a.label
                FROM standard AS s
                JOIN articles AS a ON s.article_id = a.id
                    AND a.title IN (SELECT gid FROM gids)
                ORDER BY
                s.embedding {self.dist_metric.value} %s
                LIMIT %s;
                """
            params = (str(query_id), str(query_id), vec, vec, n)
            results = self._fetch(sql, params)
            self._similarity_cache[cache_key] = results

        return [
            (dist, Article(title, content, label))
            for dist, title, content, label in results[:n]
        ]

    def edge_search(
        self,
        query_id: int,
        emb: FullEmbeddings,
        n: int = 32
    ) -> list[tuple[float, Article]]:
        """
        只在 graph_id=query_id 对应的边子集里做标准向量相似度检索。
        """
        vec = np.array(emb.standard_embedding)
        cache_key = (f"edge-{query_id}", vec.data.tobytes())
        results = self._similarity_cache.get(cache_key)
        if not results or len(results) < n:
            sql = f"""
            WITH gids AS (
            SELECT edge_gid AS gid FROM graph_triples
                WHERE graph_id = (
                SELECT id FROM graphs WHERE question_id = %s
                )
            )
            SELECT
            s.embedding {self.dist_metric.value} %s AS distance,
            a.title, a.content, a.label
            FROM standard AS s
            JOIN articles AS a ON s.article_id = a.id
                AND a.title IN (SELECT gid FROM gids)
            ORDER BY
            s.embedding {self.dist_metric.value} %s
            LIMIT %s;
            """
            params = (str(query_id), vec, vec, n)
            results = self._fetch(sql, params)
            self._similarity_cache[cache_key] = results

        return [
            (dist, Article(title, content, label))
            for dist, title, content, label in results[:n]
        ]

    def entity_attention_search(
        self,
        emb: FullEmbeddings,
        layer_idx: int,
        query_id: str,
        n: int = 32
    ) -> list[list[tuple[float, Article]]]:
        """
        对每个头，在 graph_id=query_id 对应的实体子集里做注意力空间检索。
        返回值与原 attention_search 一致：List[head_i 的 top-K 结果列表]。
        """
        results: List[List[Tuple[float, Article]]] = []
        # 查询向量：第 layer_idx 层，每个头的向量
        head_vecs = np.array(emb.layer_embeddings[layer_idx].attention_heads)

        for head_idx, vec in enumerate(head_vecs):
            cache_key = (f"ent-attn-{query_id}-{layer_idx}-{head_idx}", vec.tobytes())
            head_res = self._similarity_cache.get(cache_key, [])

            if len(head_res) < n:
                sql = f"""
                WITH gids AS (
                SELECT head_gid AS gid FROM graph_triples
                    WHERE graph_id = (
                    SELECT id FROM graphs WHERE question_id = %s
                    )
                UNION
                SELECT tail_gid FROM graph_triples
                    WHERE graph_id = (
                    SELECT id FROM graphs WHERE question_id = %s
                    )
                )
                SELECT
                att.head{head_idx:02} {self.dist_metric.value} %s AS distance,
                a.title, a.content, a.label
                FROM attention AS att
                JOIN articles AS a ON att.article_id = a.id
                    AND a.title IN (SELECT gid FROM gids)
                WHERE att.layer_index = %s
                ORDER BY
                att.head{head_idx:02} {self.dist_metric.value} %s
                LIMIT %s;
                """
                params = (str(query_id), str(query_id), vec, layer_idx, vec, n)
                head_res = self._fetch(sql, params)
                self._similarity_cache[cache_key] = head_res

            # 构造 Article 对象列表
            articles = [
                (dist, Article(title, content, label))
                for dist, title, content, label in head_res[:n]
            ]
            results.append(articles)

        return results


    def edge_attention_search(
        self,
        query_id: int,
        emb: FullEmbeddings,
        layer_idx: int,
        n: int = 32
    ) -> list[list[tuple[float, Article]]]:
        """
        对每个头，在 graph_id=query_id 对应的边子集里做注意力空间检索。
        返回值与原 attention_search 一致：List[head_i 的 top-K 结果列表]。
        """
        results: List[List[Tuple[float, Article]]] = []
        head_vecs = np.array(emb.layer_embeddings[layer_idx].attention_heads)

        for head_idx, vec in enumerate(head_vecs):
            cache_key = (f"edge-attn-{query_id}-{layer_idx}-{head_idx}", vec.tobytes())
            head_res = self._similarity_cache.get(cache_key, [])

            if len(head_res) < n:
                sql = f"""
                WITH gids AS (
                SELECT edge_gid AS gid FROM graph_triples
                    WHERE graph_id = (
                    SELECT id FROM graphs WHERE question_id = %s
                    )
                )
                SELECT
                att.head{head_idx:02} {self.dist_metric.value} %s AS distance,
                a.title, a.content, a.label
                FROM attention AS att
                JOIN articles AS a ON att.article_id = a.id
                    AND a.title IN (SELECT gid FROM gids)
                WHERE att.layer_index = %s
                ORDER BY
                att.head{head_idx:02} {self.dist_metric.value} %s
                LIMIT %s;
                """
                params = (str(query_id), vec, layer_idx, vec, n)
                head_res = self._fetch(sql, params)
                self._similarity_cache[cache_key] = head_res

            articles = [
                (dist, Article(title, content, label))
                for dist, title, content, label in head_res[:n]
            ]
            results.append(articles)

        return results