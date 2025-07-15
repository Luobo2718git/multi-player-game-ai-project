"""
人类智能体
处理人类玩家的输入
"""

import time
from typing import Dict, List, Tuple, Any, Optional, Union
from ..base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """人类智能体"""
    
    def __init__(self, name: str = "Human", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        获取人类玩家的动作
        
        Args:
            observation: 当前观察
            env: 环境对象
            
        Returns:
            人类玩家选择的动作
        """
        start_time = time.time()
        
        # 显示当前游戏状态
        self._display_game_state(observation, env)
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        # 获取人类输入
        action = self._get_human_input(valid_actions, env)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def _display_game_state(self, observation: Any, env: Any):
        """显示游戏状态"""
        print(f"\n=== {self.name} 的回合 ===")
        print(f"玩家ID: {self.player_id}")
        
        # 显示棋盘
        if hasattr(env, 'render'):
            env.render(mode='human')
        
        # 显示有效动作数量
        valid_actions = env.get_valid_actions()
        print(f"可用位置数量: {len(valid_actions)}")
    
    def _get_human_input(self, valid_actions: List[Any], env: Any) -> Any:
        """获取人类输入"""
        print("当前棋盘：")
        env.render()
        print(f"可选动作: {valid_actions}")
        while True:
            try:
                move = input(f"玩家{self.player_id}请输入落子位置(如 0,0): ")
                row, col = map(int, move.strip().split(','))
                if (row, col) in valid_actions:
                    return (row, col)
                else:
                    print("无效位置，请重新输入。")
            except Exception:
                # 根据游戏类型获取不同的输入
                if hasattr(env, 'board_size'):  # 五子棋
                    action = self._get_gomoku_input(env.board_size)
                else:
                    # 默认输入格式
                    action = self._get_default_input(valid_actions)
                
                # 验证输入
                if action in valid_actions:
                    return action
                else:
                    print(f"无效动作: {action}")
                    print("请重新输入")
                    
            except (ValueError, IndexError) as e:
                print(f"输入错误: {e}")
                print("请重新输入")
    
    def _get_gomoku_input(self, board_size: int) -> Tuple[int, int]:
        """获取五子棋输入"""
        print(f"请输入行和列 (0-{board_size-1}):")
        
        # 获取行
        while True:
            try:
                row_input = input("行: ").strip()
                if row_input.lower() == 'quit':
                    raise KeyboardInterrupt
                row = int(row_input)
                if 0 <= row < board_size:
                    break
                else:
                    print(f"行必须在 0-{board_size-1} 之间")
            except ValueError:
                print("请输入有效的数字")
        
        # 获取列
        while True:
            try:
                col_input = input("列: ").strip()
                if col_input.lower() == 'quit':
                    raise KeyboardInterrupt
                col = int(col_input)
                if 0 <= col < board_size:
                    break
                else:
                    print(f"列必须在 0-{board_size-1} 之间")
            except ValueError:
                print("请输入有效的数字")
        
        return (row, col)
    
    def _get_default_input(self, valid_actions: List[Any]) -> Any:
        """获取默认输入"""
        print("可用动作:")
        for i, action in enumerate(valid_actions):
            print(f"{i}: {action}")
        
        while True:
            try:
                choice = input("请选择动作编号: ").strip()
                if choice.lower() == 'quit':
                    raise KeyboardInterrupt
                index = int(choice)
                if 0 <= index < len(valid_actions):
                    return valid_actions[index]
                else:
                    print(f"编号必须在 0-{len(valid_actions)-1} 之间")
            except ValueError:
                print("请输入有效的数字")
    
    def reset(self):
        """重置人类智能体"""
        super().reset()
        # 人类智能体不需要特殊重置
    
    def get_info(self) -> Dict[str, Any]:
        """获取人类智能体信息"""
        info = super().get_info()
        info.update({
            'type': 'Human',
            'description': '人类玩家，通过键盘输入进行游戏'
        })
        return info 
    
class bombHumanAgent(BaseAgent):
    """Bomb游戏的人类智能体"""
    
    """
    人类玩家智能体，通过键盘输入控制。
    """
    def __init__(self, name="Human", player_id=1):
        super().__init__(name, player_id)
        self.action: Optional[Union[Tuple[int, int], Tuple[int, int, bool]]] = (0, 0) # 默认不动
        self.last_action_time = time.time()
        self.action_delay = 0.1 # 限制按键频率，防止过快输入

    def set_action(self, action: Union[Tuple[int, int], Tuple[int, int, bool]]):
        """外部设置动作（例如通过GUI的事件处理）"""
        current_time = time.time()
        if current_time - self.last_action_time > self.action_delay:
            self.action = action
            self.last_action_time = current_time

    def get_action(self, observation: Dict[str, Any], env: Any) -> Union[Tuple[int, int], Tuple[int, int, bool]]:
        """
        获取人类玩家的动作。
        这个方法主要用于返回通过 set_action 设置的动作。
        """
        # 在这里可以添加一些调试信息或游戏状态显示
        # self._display_game_state(observation, env) # 暂时注释掉，避免在每次get_action时都尝试渲染

        # 获取并返回当前设置的动作，然后重置为默认不动，等待下一次输入
        current_action = self.action
        self.action = (0, 0) # 重置为默认不动，防止连续执行
        return current_action

    def reset(self):
        """重置玩家状态"""
        self.action = (0, 0)
        self.last_action_time = time.time()

    def _display_game_state(self, observation: Dict[str, Any], env: Any):
        """
        在控制台显示简化版游戏状态 (仅用于调试)
        注意: 在实际Pygame GUI中，这部分逻辑应由GUI类处理
        """
        # 仅在调试时使用，避免与Pygame GUI冲突
        # print("\n--- 游戏状态 ---")
        # board = observation['board']
        # for r in range(board.shape[0]):
        #     row_str = ""
        #     for c in range(board.shape[1]):
        #         val = board[r, c]
        #         if (r, c) == observation['player1_pos']:
        #             row_str += "P1 "
        #         elif (r, c) == observation['player2_pos']:
        #             row_str += "P2 "
        #         elif val == BombGame.WALL:
        #             row_str += "## "
        #         elif val == BombGame.DESTRUCTIBLE_BLOCK:
        #             row_str += "DB "
        #         elif val >= BombGame.BOMB_START_ID and val < BombGame.EXPLOSION:
        #             row_str += f"B{val - BombGame.BOMB_START_ID} "
        #         elif val == BombGame.EXPLOSION:
        #             row_str += "XX "
        #         elif val == BombGame.EMPTY:
        #             row_str += ".  "
        #         elif val >= BombGame.ITEM_BOMB_UP: # 物品
        #             item_map = {
        #                 BombGame.ITEM_BOMB_UP: "B+",
        #                 BombGame.ITEM_RANGE_UP: "R+",
        #                 BombGame.ITEM_SHIELD: "S ",
        #                 BombGame.ITEM_SQUARE_RANGE_UP: "SQ"
        #             }
        #             row_str += f"{item_map.get(val, '??')} "
        #         else:
        #             row_str += f"{val:2d} "
        #     print(row_str)
        
        # 获取有效动作时，传入 player_id
        valid_actions = env.get_valid_actions(self.player_id) # <--- 关键修改在这里
        # print(f"有效动作 ({self.name}): {valid_actions}")
        # print(f"当前玩家位置: {observation['player1_pos'] if self.player_id == 1 else observation['player2_pos']}")
        # print(f"玩家 {self.player_id} 存活: {observation['alive1'] if self.player_id == 1 else observation['alive2']}")
        # if self.player_id == 1:
        #     print(f"P1 炸弹数: {observation['player1_current_bombs']}/{observation['player1_bombs_max']}, 范围: {observation['player1_range']}, 护盾: {observation['player1_shield_active']}")
        # else:
        #     print(f"P2 炸弹数: {observation['player2_current_bombs']}/{observation['player2_bombs_max']}, 范围: {observation['player2_range']}, 护盾: {observation['player2_shield_active']}")
