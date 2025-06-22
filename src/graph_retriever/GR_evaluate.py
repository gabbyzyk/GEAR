import json
import pandas as pd
import re  # 导入正则表达式库，用于复杂的字符串匹配和替换
import string # 导入字符串库，主要用于获取标点符号集合


def get_accuracy_gqa(path):
    """
    计算 GQA 数据集上的准确率。
    输入: path (str) - 存储模型预测结果的 JSONL 文件路径。
    行为: 遍历每一条预测，如果标准答案(label)字符串是预测(pred)字符串的子串，则视为正确。
    输出: (float) - 准确率 (正确数量 / 总数量)。
    """
    # 使用 pandas 读取 JSONL 文件，每行是一个 JSON 对象
    df = pd.read_json(path, lines=True)
    # 计算准确率
    correct = 0  # 初始化正确计数器
    # 遍历预测列和标签列中的每一对
    for pred, label in zip(df["pred"], df["label"]):
        # 如果标签字符串存在于预测字符串中
        if label in pred:
            correct += 1 # 计数器加一
    # 返回最终的准确率
    return correct / len(df)


def get_accuracy_expla_graphs(path):
    """
    计算 ExplaGraphs 数据集上的准确率。
    输入: path (str) - 存储模型预测结果的 JSONL 文件路径。
    行为: 使用正则表达式查找预测中的关键词("support"或"counter")，并与标签进行比较。
    输出: (float) - 准确率。
    """
    df = pd.read_json(path, lines=True)
    # 计算准确率
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        # 从预测字符串中查找所有 "support" 或 "counter" (不区分大小写)
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        # 如果找到了匹配项，并且第一个匹配项的小写形式与标签相同
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)


def normalize(s: str) -> str:
    """
    一个辅助函数，用于标准化字符串：转为小写、移除标点、冠词和多余空格。
    输入: s (str) - 原始字符串。
    行为: 对字符串进行一系列清洗操作。
    输出: (str) - 清洗和标准化后的字符串。
    """
    s = s.lower() # 转为小写
    exclude = set(string.punctuation) # 获取所有标点符号
    s = "".join(char for char in s if char not in exclude) # 移除所有标点
    s = re.sub(r"\b(a|an|the)\b", " ", s) # 移除冠词 "a", "an", "the"
    # 移除 <pad> 符号
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split()) # 移除多余的空格
    return s


def match(s1: str, s2: str) -> bool:
    """
    一个辅助函数，用于判断一个字符串是否是另一个字符串的子串（在标准化之后）。
    输入: s1 (str), s2 (str) - 两个待比较的字符串。
    行为: 先对两个字符串进行标准化，然后检查 s2 是否在 s1 中。
    输出: (bool) - 如果 s2 是 s1 的子串，则返回 True，否则返回 False。
    """
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    """
    一个辅助函数，用于计算 F1 分数、精确率和召回率。
    输入: prediction (list[str]), answer (list[str]) - 预测的答案列表和标准答案列表。
    行为: 计算预测列表中有多少项与标准答案列表匹配。
    输出: (tuple) - (f1分数, 精确率, 召回率)。
    """
    # 如果预测为空，则所有指标都为0
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    # 将预测列表拼接成一个长字符串，以便使用 match 函数
    prediction_str = " ".join(prediction)
    # 遍历每一个标准答案
    for a in answer:
        # 如果标准答案能在预测字符串中找到
        if match(prediction_str, a):
            matched += 1
    # 计算精确率: 匹配上的数量 / 预测的总数
    precision = matched / len(prediction)
    # 计算召回率: 匹配上的数量 / 标准答案的总数
    recall = matched / len(answer)
    # 如果精确率和召回率都为0，则F1也为0，避免除零错误
    if precision + recall == 0:
        return 0, precision, recall
    else:
        # 计算F1分数
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    """
    一个辅助函数，用于计算 "soft" 准确率。
    输入: prediction (str), answer (list[str]) - 预测字符串和标准答案列表。
    行为: 计算标准答案列表中有多少项被预测字符串所覆盖。
    输出: (float) - 匹配上的标准答案占总标准答案的比例。
    """
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    """
    一个辅助函数，用于计算命中率 (Hit Rate)。
    输入: prediction (str), answer (list[str]) - 预测字符串和标准答案列表。
    行为: 检查标准答案列表中是否至少有一个答案出现在预测中。
    输出: (int) - 只要命中一个就返回1，否则返回0。
    """
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_webqsp(path):
    """
    计算 WebQSP 数据集上的多种评估指标 (F1, Hit Rate, Accuracy等)。
    输入: path (str) - 存储模型预测结果的 JSONL 文件路径。
    行为: 遍历所有样本，对每个样本计算F1, Accuracy, Hit等指标，最后求平均值并打印。
    输出: (float) - 命中率 (Hit Rate)。
    """
    df = pd.read_json(path, lines=True)

    # 初始化用于存储每个样本得分的列表
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    # 遍历预测列和标签列
    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):

        # WebQSP 的答案/预测可能用'|'分隔，需要预处理
        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")
        prediction = prediction.split("\n")

        # 调用辅助函数计算各项指标
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        # 将得分添加到各自的列表中
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)

        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    # 计算所有样本的平均分，并乘以100转为百分比
    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    # 打印所有指标的最终结果
    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # 将命中率作为函数的主要返回值
    return hit


# --- 评估函数分派器 ---
# 创建一个字典，将数据集名称映射到其对应的评估函数
eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp,
    "webqsp_baseline": get_accuracy_webqsp,
}