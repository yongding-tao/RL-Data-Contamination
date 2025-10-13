import torch
import re
import sys

def extract_answer(response_str):
    """
    从模型的响应字符串中提取最终答案。
    专门查找并返回最后一个 \\boxed{} 中的内容。

    Args:
        response_str (str): 模型的完整输出字符串。

    Returns:
        str or None: 如果找到，则返回提取的答案；否则返回 None。
    """
    try:
        matches = re.findall(r'\\boxed{(.*?)}', response_str)
        if matches:
            return matches[-1]  # 返回最后一个匹配项
    except Exception as e:
        # 在正则表达式出现问题时打印错误
        print(f"Error during answer extraction: {e}")
    return None

def calc_sat_value(clause, solution):
    """
    计算一个解（solution）是否满足一个逻辑子句（clause）。

    Args:
        clause (str): 表示逻辑表达式的字符串，例如 'A & !B'。
        solution (str): 表示变量赋值的二进制字符串，例如 '10'。

    Returns:
        int: 如果解满足子句，则返回 1，否则返回 0。
    """
    def parse_literals(clause_str):
        literals = []
        i = 0
        while i < len(clause_str):
            if clause_str[i] == '!':
                literals.append(clause_str[i:i+2])
                i += 2
            else:
                literals.append(clause_str[i])
                i += 1
        return literals
    
    for subclause in clause.split(' & '):
        satisfied = False
        for lit in parse_literals(subclause):
            neg = False
            if lit.startswith('!'):
                var = lit[1]
                neg = True
            else:
                var = lit
            
            idx = ord(var) - ord('A')
            if idx >= len(solution):
                val = '0'  # 如果解的长度不够，则假定未指定的变量为 '0'
            else:
                val = solution[idx]
            
            if (neg and val == '0') or (not neg and val == '1'):
                satisfied = True
                break
        
        if not satisfied:
            return 0  # 任何一个子句不满足，则整个表达式不满足
    return 1  # 所有子句都满足

def compute_score(solution_str, clause):
    """
    为单个样本计算分数。

    Args:
        solution_str (str): 模型的完整输出。
        ground_truth (str): 期望的真值标签（逻辑子句）。

    Returns:
        float: 计算出的分数。
    """
    predicted_answer = extract_answer(solution_str)
    # print('solution_str: ', solution_str)
    # print("predicted_answer: ", predicted_answer)
    
    if predicted_answer is None:
        return -1  # 如果没有找到答案，返回格式分数（或0）
    
    # 检查解是否满足逻辑子句
    if calc_sat_value(clause, predicted_answer):
        return 1  # 答案正确
    else:
        return 0 # 答案错误