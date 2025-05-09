"""
Werewolf Game Belief State Visualization Tool
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Any, Tuple
import io
from PIL import Image

from werewolf_env.state import GameState
from utils.belief_updater import BeliefState, RoleSpecificBeliefUpdater


class BeliefVisualizer:
    """Belief state visualization class"""
    
    @staticmethod
    def plot_belief_heatmap(belief_state: BeliefState) -> Image.Image:
        """
        Plot belief state heatmap
        
        Args:
            belief_state: Belief state object
            
        Returns:
            Heatmap image
        """
        beliefs = belief_state.beliefs
        if not beliefs:
            # Create an empty image
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No belief data", horizontalalignment='center',
                   verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        
        # Collect all players and roles
        players = list(beliefs.keys())
        all_roles = set()
        for player_beliefs in beliefs.values():
            all_roles.update(player_beliefs.keys())
        all_roles = sorted(list(all_roles))
        
        # Create data matrix
        data = np.zeros((len(players), len(all_roles)))
        for i, player_id in enumerate(players):
            for j, role in enumerate(all_roles):
                data[i, j] = beliefs[player_id].get(role, 0.0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap=cm.Blues)
        
        # Add axis labels
        ax.set_xticks(np.arange(len(all_roles)))
        ax.set_yticks(np.arange(len(players)))
        ax.set_xticklabels(all_roles, rotation=45, ha="right")
        ax.set_yticklabels([f"Player {pid}" for pid in players])
        
        # Add value labels
        for i in range(len(players)):
            for j in range(len(all_roles)):
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                              ha="center", va="center", color="black" if data[i, j] < 0.7 else "white")
        
        # Add colorbar and title
        plt.colorbar(im)
        ax.set_title("Belief State Heatmap")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def plot_certain_roles(belief_state: BeliefState) -> Image.Image:
        """
        Plot certain role information
        
        Args:
            belief_state: Belief state object
            
        Returns:
            Image
        """
        certain_roles = belief_state.certain_roles
        
        if not certain_roles:
            # Create an empty image
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No certain role information", horizontalalignment='center',
                   verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        players = list(certain_roles.keys())
        roles = [certain_roles[pid] for pid in players]
        
        # Assign different colors to different roles
        unique_roles = set(roles)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_roles)))
        role_colors = {role: colors[i] for i, role in enumerate(unique_roles)}
        bar_colors = [role_colors[role] for role in roles]
        
        ax.bar(range(len(players)), [1] * len(players), color=bar_colors, tick_label=[f"Player {pid}" for pid in players])
        
        # Add role labels
        for i, (player, role) in enumerate(zip(players, roles)):
            ax.text(i, 0.5, role, ha='center', va='center', rotation=0, fontsize=10)
        
        # Add legend
        patches = [plt.Rectangle((0, 0), 1, 1, color=role_colors[role]) for role in unique_roles]
        ax.legend(patches, unique_roles, loc='upper right')
        
        ax.set_title("Certain Role Information")
        ax.set_ylabel("Role")
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def plot_most_likely_roles(belief_state: BeliefState) -> Image.Image:
        """
        Plot most likely role for each player
        
        Args:
            belief_state: Belief state object
            
        Returns:
            Image
        """
        beliefs = belief_state.beliefs
        
        if not beliefs:
            # Create an empty image
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No belief data", horizontalalignment='center',
                   verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        
        # Get most likely role for each player
        most_likely = {}
        for player_id in beliefs:
            role, prob = belief_state.get_most_likely_role(player_id)
            most_likely[player_id] = (role, prob)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        players = list(most_likely.keys())
        probs = [most_likely[pid][1] for pid in players]
        roles = [most_likely[pid][0] for pid in players]
        
        # Assign different colors to different roles
        unique_roles = set(roles)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_roles)))
        role_colors = {role: colors[i] for i, role in enumerate(unique_roles)}
        bar_colors = [role_colors[role] for role in roles]
        
        ax.bar(range(len(players)), probs, color=bar_colors, tick_label=[f"Player {pid}" for pid in players])
        
        # Add role labels
        for i, (player, (role, prob)) in enumerate(zip(players, [most_likely[pid] for pid in players])):
            ax.text(i, prob / 2, role, ha='center', va='center', rotation=0, fontsize=10)
        
        # Add legend
        patches = [plt.Rectangle((0, 0), 1, 1, color=role_colors[role]) for role in unique_roles]
        ax.legend(patches, unique_roles, loc='upper right')
        
        ax.set_title("Most Likely Roles")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def plot_werewolf_probabilities(belief_state: BeliefState) -> Image.Image:
        """
        Plot probability of each player being a werewolf
        
        Args:
            belief_state: Belief state object
            
        Returns:
            Image
        """
        beliefs = belief_state.beliefs
        
        if not beliefs:
            # Create an empty image
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No belief data", horizontalalignment='center',
                   verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        
        # Get werewolf probability for each player
        werewolf_probs = {}
        for player_id in beliefs:
            werewolf_probs[player_id] = beliefs[player_id].get('werewolf', 0.0)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        players = list(werewolf_probs.keys())
        probs = [werewolf_probs[pid] for pid in players]
        
        ax.bar(range(len(players)), probs, tick_label=[f"Player {pid}" for pid in players])
        
        # Add probability labels
        for i, (player, prob) in enumerate(zip(players, probs)):
            ax.text(i, prob / 2, f"{prob:.2f}", ha='center', va='center', rotation=0, fontsize=10)
        
        ax.set_title("Werewolf Probabilities")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def plot_game_state(game_state: GameState, player_id: int) -> Image.Image:
        """
        绘制游戏状态概览
        
        Args:
            game_state: 游戏状态
            player_id: 视角玩家ID
            
        Returns:
            图像
        """
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制玩家角色信息
        player_roles = []
        for i, player in enumerate(game_state.players):
            if i == player_id:
                # 玩家自己的角色信息是完整的
                player_roles.append((i, player['original_role'], player['current_role']))
            else:
                # 其他玩家的角色信息未知
                player_roles.append((i, "???", "???"))
        
        # 绘制角色分布
        y_pos = np.arange(len(player_roles))
        
        # 绘制原始角色
        original_roles = [info[1] for info in player_roles]
        ax1.barh(y_pos, [1] * len(player_roles), color='skyblue', alpha=0.6)
        
        # 添加角色标签
        for i, (pid, original, current) in enumerate(player_roles):
            ax1.text(0.5, i, original, va='center', ha='center')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"Player {info[0]}" for info in player_roles])
        ax1.set_title('Original Role Distribution')
        ax1.set_xlim(0, 1)
        
        # 绘制当前角色
        current_roles = [info[2] for info in player_roles]
        ax2.barh(y_pos, [1] * len(player_roles), color='lightgreen', alpha=0.6)
        
        # 添加角色标签
        for i, (pid, original, current) in enumerate(player_roles):
            ax2.text(0.5, i, current, va='center', ha='center')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"Player {info[0]}" for info in player_roles])
        ax2.set_title('Current Role Distribution')
        ax2.set_xlim(0, 1)
        
        # 添加游戏状态信息
        fig.suptitle(f"Game State - Phase: {game_state.phase}, Round: {game_state.round}", fontsize=16)
        
        # 添加中央牌堆信息
        if game_state.phase == 'end':
            # 游戏结束时显示中央牌堆
            center_text = "Central Card Pile: " + ", ".join(game_state.center_cards)
            fig.text(0.5, 0.01, center_text, ha='center')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 将图像转换为PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def generate_belief_report(belief_updater: RoleSpecificBeliefUpdater, game_state: GameState) -> Dict[str, Image.Image]:
        """
        Generate a comprehensive belief report
        
        Args:
            belief_updater: Belief updater object
            game_state: Game state object
            
        Returns:
            Dictionary of visualization images
        """
        belief_state = belief_updater.belief_state
        player_id = belief_updater.player_id
        
        report = {
            'belief_heatmap': BeliefVisualizer.plot_belief_heatmap(belief_state),
            'certain_roles': BeliefVisualizer.plot_certain_roles(belief_state),
            'most_likely': BeliefVisualizer.plot_most_likely_roles(belief_state),
            'werewolf_probs': BeliefVisualizer.plot_werewolf_probabilities(belief_state),
            'game_state': BeliefVisualizer.plot_game_state(game_state, player_id)
        }
        
        return report 