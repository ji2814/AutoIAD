from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

def calculate_llm_cost(prompt_tokens: int, completion_tokens: int,
                       prompt_cost_per_million: float, completion_cost_per_million: float) -> float:
    """
    计算 LLM 调用的总成本。

    参数:
    - prompt_tokens: prompt 部分的 token 数量
    - completion_tokens: completion 部分的 token 数量
    - prompt_cost_per_million: 每百万 prompt tokens 的费用（单位：美元）
    - completion_cost_per_million: 每百万 completion tokens 的费用（单位：美元）

    返回:
    - 总成本（单位：美元，float 类型）
    """
    cost = (prompt_tokens / 1_000_000) * prompt_cost_per_million + \
           (completion_tokens / 1_000_000) * completion_cost_per_million
    return round(cost, 6)  # 保留小数点后 6 位，精度足够高

import time
import threading
from functools import wraps

def timeout(seconds: int = 6):
    """
    用于超时控制的装饰器
    :param seconds: 超时时间，单位为秒，默认为600秒(10分钟)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 用于存储函数结果的变量
            result = []
            error = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            
            # 创建并启动线程
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            # 等待指定时间
            thread.join(seconds)
            
            # 检查线程是否仍在运行
            if thread.is_alive():
                raise TimeoutError(f"函数 {func.__name__} 运行时间超过了 {seconds} 秒")
            
            # 检查是否有异常发生
            if error:
                raise error[0]
            
            # 返回函数结果
            return result[0]
        
        return wrapper
    return decorator