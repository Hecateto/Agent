import ast
import operator
import math
from hello_agents import ToolRegistry

def my_calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        str: The result of the evaluation or an error message.
    """
    if not expression.strip():
        return "Error: The expression is empty."

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    functions = {
        'sqrt': math.sqrt,
        'log': math.log,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
    }

    try:
        node = ast.parse(expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def _eval_node(node, operators, functions):
    """ Recursively evaluate an AST node. """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op), None)
        return op(left, right)
    elif isinstance(node, ast.Call):
        func = node.func.id
        if func in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]

def create_calculator_registry():
    """
    Creates a ToolRegistry with the my_calculate tool.

    Returns:
        ToolRegistry: The registry containing the my_calculate tool.
    """
    registry = ToolRegistry()
    registry.register_function(
        name="my_calculator",
        description="简单的计算器，支持基本的数学运算和函数调用。",
        func=my_calculate
    )
    return registry



def test_calculator_tool():
    """测试自定义计算器工具"""

    registry = create_calculator_registry()

    print("🧪 测试自定义计算器工具\n")

    test_cases = [
        "2 + 3",           # 基本加法
        "10 - 4",          # 基本减法
        "5 * 6",           # 基本乘法
        "15 / 3",          # 基本除法
        "sqrt(16)",        # 平方根
        "log(100)",       # 对数
        "sin(0)",         # 正弦函数
        "cos(0)",         # 余弦函数
        "tan(45)",        # 正切函数
        "2 ** 3",         # 幂运算
        "invalid_expr",   # 无效表达式
        "",                # 空表达式
        "10 / 0"          # 除以零
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("my_calculator", expression)
        print(f"结果: {result}\n")

def test_with_simple_agent():
    """测试与SimpleAgent的集成"""
    from my_llm import MyLLM

    llm = MyLLM()
    registry = create_calculator_registry()
    print("🤖 与SimpleAgent集成测试:")

    user_question = "请帮我计算 sqrt(16) + 2 * 3"

    print(f"用户问题: {user_question}")

    calc_result = registry.execute_tool("my_calculator", "sqrt(16) + 2 * 3")
    print(f"计算结果: {calc_result}")

    final_messages = [
        {"role": "user", "content": f"计算结果是 {calc_result}，请用自然语言回答用户的问题:{user_question}"}
    ]

    print("\n🎯 SimpleAgent的回答:")
    response = llm.invoke(final_messages).strip()
    print(response)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_calculator_tool()
    test_with_simple_agent()
