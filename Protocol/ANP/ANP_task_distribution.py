from hello_agents.protocols import ANPDiscovery, register_service
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools.builtin import ANPTool
import random
from dotenv import load_dotenv

load_dotenv()
llm = HelloAgentsLLM()

# 创建 ANP 服务发现实例
discovery = ANPDiscovery()

# 注册多个计算节点服务
NODES = 10
for i in range(NODES):
    register_service(
        discovery=discovery,
        service_id=f"compute_node_{i}",
        service_name=f"计算节点{i}",
        service_type='computation',
        capabilities=['data_processing', 'model_training'],
        endpoint=f'https://node{i}:8000',
        metadata={
            "load": random.uniform(0.1, 0.9),
            "cpu_cores": random.choice([4, 8, 16]),
            "memory_gb": random.choice([16, 32, 64]),
            "gpu": random.choice([True, False])
        }
    )

print(f" ✅ 注册了 {len(discovery.list_all_services())} 个计算节点")
print("计算节点列表:")
for service in discovery.list_all_services():
    print(f"- {service.service_id}: {service.metadata}")

# 创建一个使用 ANP 的智能体
scheduler = SimpleAgent(
    name="ANP调度智能体",
    llm=llm,
    system_prompt="""
    你是一个智能任务调度器，负责：
    1. 分析任务需求
    2. 选择最合适的计算节点
    3. 分配任务

    选择节点时考虑：负载、CPU核心数、内存、GPU等因素。

    使用 service_discovery 工具时，必须提供 action 参数：
    - 查看所有节点：{"action": "discover_services", "service_type": "computation"}
    - 获取网络统计：{"action": "get_stats"}
    """
)

anp_tool = ANPTool(
    name='service_discovery',
    description='服务发现工具, 用于查找和选择计算节点',
    discovery=discovery
)
scheduler.add_tool(anp_tool)


def assign_task(task_description: str):
    prompt = f"""
    任务描述: {task_description}
    请分析任务需求，选择最合适的计算节点，并分配任务。
    步骤：
    1. 使用 service_discovery 工具查看所有可用的计算节点（service_type="computation"）
    2. 分析每个节点的特点（负载、CPU核心数、内存、GPU等）
    3. 根据任务需求选择最合适的节点
    4. 说明选择理由
    请直接给出最终选择的节点ID和理由。
    """
    response = scheduler.run(prompt)
    print("调度结果:")
    print(response)


if __name__ == "__main__":
    sample_tasks = [
        "需要处理一个大规模的数据集，进行复杂的模型训练，要求使用GPU加速。",
        "需要进行简单的数据清洗和统计分析，计算资源需求较低。",
        "需要运行一个需要大量内存的仿真程序，CPU核心数要求较高."
    ]
    for task in sample_tasks:
        print(f"\n分配任务: {task}")
        assign_task(task)

