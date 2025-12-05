# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法
"""
import asyncio
import os
import random
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.formatter import OpenAIMultiAgentFormatter

from game_roles import GameRoles
from prompt import ChinesePrompts
from structured_output import (
    DiscussionModelCN,
    WitchActionModelCN,
    WerewolfKillModelCN,
    get_seer_model_cn,
    get_vote_model_cn,
    get_hunter_model_cn
)
from utils import (
    get_chinese_name,
    format_player_list,
    majority_vote_cn,
    check_winning_cn,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND
)


class ThreeKingdomsWerewolfGame:
    """ 三国狼人杀游戏主类 """
    def __init__(self):
        self.players: Dict[str, ReActAgent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[ReActAgent] = []
        self.werewolves: List[ReActAgent] = []
        self.villagers: List[ReActAgent] = []
        self.seer: List[ReActAgent] = []
        self.witch: List[ReActAgent] = []
        self.hunter: List[ReActAgent] = []

        self.witch_has_antidote = True
        self.witch_has_poison = True

    async def create_player(self, role: str, character: str) -> ReActAgent:
        """ 创建三国玩家 """
        name = get_chinese_name(character)
        self.roles[name] = role

        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=OpenAIChatModel(
                model_name=os.getenv("MODEL"),
                api_key=os.getenv("API_KEY"),
                client_args={
                    "base_url": os.getenv("BASE_URL")
                },
                generate_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
            ),
            formatter=OpenAIMultiAgentFormatter()
        )

        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )
        self.players[name] = agent
        return agent

    async def setup_game(self, num_players: int=6):
        """ 设置游戏 """
        print("🎮 开始设置三国狼人杀游戏...")

        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], num_players)
        roles = GameRoles.get_standard_setup(num_players)

        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)

            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print("🎮 游戏设置完成！")

    async def werewolf_phase(self, round_num: int):
        """ 狼人阶段 """
        if not self.werewolves:
            return None

        await self.moderator.announce(f"🌙 第{round_num}轮：狼人请睁眼，选择要杀死的目标。")

        async with MsgHub(
            self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"狼人们，请讨论并选择一名玩家进行杀害。存活玩家：{format_player_list(self.alive_players)}"
            )
        ) as werewolves_hub:
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)
            werewolves_hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请狼人们投票选择要杀死的玩家。"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False
            )

            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.werewolves[i].name] = vote_msg.metadata.get('target')
                else:
                    print(f"⚠️ 警告：狼人 {self.werewolves[i].name} 的投票无效，将随机选择目标。")
                    import random
                    valid_targets = [p.name for p in self.alive_players if p.name not in [w.name for w in self.werewolves]]
                    votes[self.werewolves[i].name] = random.choice(valid_targets) if valid_targets else None

            killed_player, _ = majority_vote_cn(votes)
            return killed_player

    async def seer_phase(self):
        """ 预言家阶段 """
        if not self.seer:
            return

        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")
        check_result = await  seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 警告：预言家 {seer_agent.name} 的查验无效，跳过查验。")
            return

        target_name = check_result.metadata.get('target')
        if not target_name:
            print(f"⚠️ 警告：预言家 {seer_agent.name} 未选择有效目标，跳过查验。")
            return
        target_role = self.roles.get(target_name, "村民")
        result_msg = f"【预言家查验结果】{target_name} 的身份是 {'狼人' if target_role == '狼人' else '好人'}。"
        await seer_agent.observe(await self.moderator.announce(result_msg))

    async def witch_phase(self, killed_player: str):
        """ 女巫阶段 """
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼，选择是否使用解药或毒药...")

        death_info = f"今晚被杀的玩家是【{killed_player}】。" if killed_player else "今晚平安无事。"
        await witch_agent.observe(await self.moderator.announce(death_info))

        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player, poisoned_player = None, None

        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 警告：女巫 {witch_agent.name} 的行动无效，跳过女巫阶段。")
        else:
            if witch_action.metadata.get('use_antidote') and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(await self.moderator.announce(f"女巫使用了解药，救活了【{killed_player}】。"))
            if witch_action.metadata.get('use_poison') and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get('target_name')
                if poisoned_player:
                    self.witch_has_poison = False
                    await witch_agent.observe(await self.moderator.announce(f"女巫使用了毒药，毒死了【{poisoned_player}】。"))

        final_killed = killed_player if not saved_player else None

        return final_killed, poisoned_player

    async def hunter_phase(self, shot_by_hunter: str):
        """ 猎人阶段 """
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")
            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 警告：猎人 {hunter_agent.name} 的行动无效，跳过猎人阶段。")
                return None

            if hunter_action.metadata.get('shoot'):
                target = hunter_action.metadata.get('target')
                if target:
                    await self.moderator.announce(f"猎人【{hunter_agent.name}】开枪带走了【{target}】。")
                    return target
                else:
                    print(f"⚠️ 警告：猎人 {hunter_agent.name} 未选择有效目标，跳过猎人阶段。")
                    return None
        return None

    def update_alive_players(self, dead_players: List[str]):
        """ 更新存活玩家列表 """
        for dead in dead_players:
            if dead in self.roles:
                role = self.roles[dead]
                agent = self.players[dead]
                if agent in self.alive_players:
                    self.alive_players.remove(agent)

                if role == "狼人" and agent in self.werewolves:
                    self.werewolves.remove(agent)
                elif role == "预言家" and agent in self.seer:
                    self.seer.remove(agent)
                elif role == "女巫" and agent in self.witch:
                    self.witch.remove(agent)
                elif role == "猎人" and agent in self.hunter:
                    self.hunter.remove(agent)
                elif agent in self.villagers:
                    self.villagers.remove(agent)

    async def day_phase(self, round_num: int):
        """ 白天讨论与投票阶段 """
        await self.moderator.announce(f"🌞 第{round_num}轮：白天开始，存活玩家请讨论并投票。")

        async with MsgHub(
            self.alive_players,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"存活玩家：{format_player_list(self.alive_players)}，请开始讨论。"
            )
        ) as day_hub:
            await sequential_pipeline(self.alive_players)
            day_hub.set_auto_broadcast(False)
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要处决的玩家。"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False
            )

            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.alive_players[i].name] = vote_msg.metadata.get('vote')
                else:
                    print(f"⚠️ 警告：玩家 {self.alive_players[i].name} 的投票无效，视为弃票。")
                    votes[self.alive_players[i].name] = None

            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)
            return voted_out

    async def play_game(self):
        """ 运行游戏主循环 """
        try:
            await self.setup_game()
            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🎲 === 第{round_num}轮游戏开始 ===")

                await self.moderator.night_announcement(round_num)

                killed_player = await self.werewolf_phase(round_num)
                await self.seer_phase()
                final_killed, poisoned_player = await self.witch_phase(killed_player)

                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                await self.moderator.death_announcement(night_deaths)

                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                voted_out = await self.day_phase(round_num)
                hunter_shot = await self.hunter_phase(voted_out)

                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(f"🎲 === 第{round_num}轮游戏结束 ===\n 存活玩家：{format_player_list(self.alive_players)}")
        except Exception as e:
            print(f"❌ 游戏过程中出现错误：{e}")

async def main():
    print("🎮 欢迎来到三国狼人杀！")
    game = ThreeKingdomsWerewolfGame()
    await game.play_game()

if __name__ == "__main__":
    asyncio.run(main())








