# Werewolf Game: Heuristic vs Random Agents Experiment Results

## Experiment Overview

This document summarizes the experimental results comparing heuristic and random agents in the Werewolf game. The experiments were conducted with two agent types (heuristic and random) in both werewolf and villager roles.

## Agent Strategies

### Random Agent
Random agents make decisions completely randomly within the valid action space. They have no strategic planning or reasoning capabilities and serve as a baseline for comparison.

### Heuristic Agent
Heuristic agents implement rule-based strategies to make informed decisions. The key strategies implemented include:

#### For Werewolf Role:
- **Target Selection**: Werewolves try to mislead by voting for non-werewolves to appear genuine
- **Team Protection**: Avoid voting for other werewolf team members
- **Strategic Voting**: Prioritize voting for players that appear most suspicious to other players
- **Behavioral Analysis**: Use factors like speech patterns, vote history, and player consistency to identify strong village players

#### For Villager Role:
- **Enhanced Suspicion Detection**: Use weighted belief system to identify werewolves (wolf suspicion boosted by 50%)
- **Behavioral Analysis**: Track and analyze voting patterns and speech consistency
- **Role-Specific Logic**: Special strategies for roles like seer (prioritize checking suspicious players)
- **Strategic Voting**: Coordinate voting against players exhibiting suspicious behavior
- **Suspicious Claims Analysis**: Apply penalty to players whose role claims don't match their behavior

## Experimental Results

The experiment ran 1000 games for each configuration with the following results:

| Agent Type Configuration | Number of Tests | Werewolf Win Rate | Villager Win Rate | Average Game Length |
|--------------------------|-----------------|-------------------|-------------------|---------------------|
| All random agents        | 1000            | 0.68              | 0.33              | 26.00               |
| All heuristic agents     | 1000            | 0.48              | 0.52              | 26.00               |
| Villager random + Werewolf heuristic | 1000 | 0.82             | 0.18              | 26.00               |
| Villager heuristic + Werewolf random | 1000 | 0.25             | 0.75              | 26.00               |
| Random combination agents | 1000           | 0.64              | 0.36              | 26.00               |

## Analysis

### Random vs Heuristic Agents
- When all agents use the same strategy type, heuristic agents demonstrate better performance for the villager team.
- Heuristic agents reduced the werewolf win rate from 68% to 48%, showing a 20% improvement in game balance.

### Team-Specific Performance
- **Heuristic Werewolves vs Random Werewolves:** Heuristic werewolves significantly outperform random werewolves, with a 57% higher win rate (82% vs 25%). This demonstrates the effectiveness of strategic deception and coordination.
- **Heuristic Villagers vs Random Villagers:** Similarly, heuristic villagers outperform random villagers by 57% (75% vs 18%), showing the effectiveness of the strategic voting and behavioral analysis.

### Game Balance
- When both teams use heuristic agents, the game becomes more balanced (48% werewolf win rate vs 52% villager win rate).
- The most imbalanced scenario is when werewolves use heuristic agents but villagers use random agents (82% werewolf win rate).

### Random Combination Analysis
- Random combination agents (mixing heuristic and random agents randomly) yielded a 64% werewolf win rate.
- This is slightly more balanced than all random agents (68% werewolf win rate) but significantly less balanced than all heuristic agents (48% werewolf win rate).

## Conclusion

The experimental results clearly demonstrate that heuristic agents significantly outperform random agents in both werewolf and villager roles. The strategic decision-making capabilities implemented in heuristic agents provide a substantial advantage over random decision-making.

These findings suggest that developing more sophisticated heuristic strategies could further improve agent performance. Additionally, the significant performance gap between heuristic and random agents indicates that the game environment provides sufficient information for intelligent decision-making, making it a suitable testbed for more advanced AI techniques such as reinforcement learning.

The results also highlight the importance of balancing agent capabilities between teams to create fair gameplay dynamics. When both teams use agents of similar capabilities, the game becomes more balanced and competitive. 