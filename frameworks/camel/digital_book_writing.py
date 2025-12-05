import os

from camel.types import ModelPlatformType
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["OPENAI_API_BASE_URL"] = os.getenv("BASE_URL")

from colorama import Fore
from camel.societies import RolePlaying
from camel.utils import print_text_animated

TASK_PROMPT = """
我们（心理学家和作家）需要合作撰写一本关于《拖延症心理学》的短篇电子书（约8000字）。
目标读者是对心理学感兴趣的普通大众。

【角色定义】
- 我（User）是心理学家：负责提供核心理论（如时间折扣、执行功能）、实证研究支持，并审核内容的科学准确性。
- 你（Assistant）是作家：负责构建书籍框架，将我提供的晦涩理论转化为生动、通俗的文字（使用比喻、案例），并打磨语言风格。

【协作流程（必须严格按阶段进行）】

阶段一：框架搭建 (第 1-5 轮)
1. 我们的第一次交互，请由你（作家）先提出一份基于心理学逻辑的完整电子书目录大纲。
2. 我（心理学家）会从专业角度审查大纲，指出遗漏的理论模块，直到我们对结构达成共识。

阶段二：核心内容生成 (第 6-20 轮)
1. 进入写作阶段后，我会逐章为你提供“硬核”知识点（例如：解释“杏仁体劫持”如何导致拖延）。
2. 你需要将这些知识点“翻译”成大众爱看的故事或比喻（例如：把杏仁体比作报警器）。
3. 请不要一次性写完，我们一节一节地推进。

阶段三：迭代与核查 (第 21-25 轮)
1. 内容完成后，你需要重新审视全书的流畅度。
2. 我会进行“事实核查”，确保你在通过比喻简化概念时，没有曲解科学原意。

阶段四：收尾 (最后几轮)
1. 总结全书的实用建议。
2. 输出最终定稿。

【交互规则】
- 每次回复请标明当前处于哪个阶段。
- 你的回复字数控制在 400 字以内，保持高频互动。
- 如果我提供的理论太难懂，请随时向我提问，而不是瞎编。
"""


if __name__ == "__main__":

    print(Fore.YELLOW + f"=== 数字书籍创作任务 ===" + Fore.RESET)
    role_play_session = RolePlaying(
        assistant_role_name="作家",
        user_role_name="心理学家",
        task_prompt=TASK_PROMPT,
        with_task_specify=False,
        model=os.getenv("MODEL"),
    )
    print(Fore.CYAN + f"=== 任务描述 ===" + Fore.RESET)
    print_text_animated(TASK_PROMPT)

    chat_turn_limit, i = 30, 0
    input_msg = role_play_session.init_chat()

    while i < chat_turn_limit:
        i += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        print(Fore.GREEN + f"\n=== 心理学家 [回合 {i}] ===\n" + Fore.RESET)
        print(user_response.msg.content)
        print(Fore.BLUE + f"\n=== 作家 [回合 {i}] ===\n" + Fore.RESET)
        print(assistant_response.msg.content)


        if "CAMEL_TASK_DONE" in user_response.msg.content:
            print(Fore.MAGENTA + "\n=== 任务完成，结束对话 ===" + Fore.RESET)
            break

        input_msg = assistant_response.msg

    print(Fore.YELLOW + f"\n=== 数字书籍创作任务结束 ===" + Fore.RESET)