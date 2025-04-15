"""
Agent factory module
创建各种类型的AI代理
"""
import logging
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

# 设置日志记录
logger = logging.getLogger("agent_factory")

def create_agent(agent_type, **kwargs):
    """
    创建AI代理
    
    Args:
        agent_type: 代理类型，如"random"或"heuristic"
        **kwargs: 创建代理所需的其他参数
        
    Returns:
        BaseAgent: 创建的AI代理实例
    """
    logger.info(f"创建代理: 类型={agent_type}, 参数={kwargs}")
    
    try:
        if agent_type == "random":
            logger.info("创建随机代理")
            agent = RandomAgent(**kwargs)
            logger.info(f"随机代理创建成功: {agent}")
            return agent
        elif agent_type == "heuristic":
            logger.info("创建启发式代理")
            agent = HeuristicAgent(**kwargs)
            logger.info(f"启发式代理创建成功: {agent}")
            return agent
        else:
            error_msg = f"不支持的代理类型: {agent_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"创建代理失败: {str(e)}")
        logger.exception(e)
        raise 