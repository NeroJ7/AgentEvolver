import functools
from typing import Type, Dict, Callable, Any

class RewardCalculatorManager:
    """
    一个单例类，用于管理和实例化不同的奖励计算器。

    该类维护一个从名称到计算器类的注册表，并提供一个装饰器
    用于自动注册，以及一个工厂方法用于根据名称获取实例。
    """
    _instance = None
    _registry: Dict[str, Type] = {}

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式。如果实例不存在，则创建一个新实例；否则，返回现有实例。
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reg(self, name: str) -> Callable:
        """
        注册装饰器。将一个类与一个给定的名称关联起来。

        用法:
            @calculator_manager.reg("my_calculator")
            class MyCalculator:
                ...
        """
        def decorator(calculator_cls: Type) -> Type:
            if name in self._registry:
                print(f"警告：名称 '{name}' 已被注册，将被新的类 '{calculator_cls.__name__}' 覆盖。")
            
            # 将类本身存入注册表
            self._registry[name] = calculator_cls
            
            # 装饰器返回原始类，不进行任何修改
            return calculator_cls
        return decorator

    def get_calculator(self, name: str, *args, **kwargs) -> Any:
        """
        工厂方法。根据注册的名称获取一个计算器类的实例。

        :param name: 注册时使用的字符串名称。
        :param args: 传递给计算器类构造函数的定位参数。
        :param kwargs: 传递给计算器类构造函数的关键字参数。
        :return: 对应计算器类的一个新实例。
        :raises ValueError: 如果提供的名称没有被注册。
        """
        calculator_cls = self._registry.get(name)
        if not calculator_cls:
            raise ValueError(f"错误：没有找到名为 '{name}' 的奖励计算器。可用名称: {list(self._registry.keys())}")
        
        # 创建并返回该类的一个新实例，并传递所有参数
        return calculator_cls(*args, **kwargs)

# ----------------------------------------------------------------------------
# 示例用法
# ----------------------------------------------------------------------------

# 1. 创建管理器单例。在整个应用程序中，您都应该使用这同一个实例。
grader_manager = RewardCalculatorManager()


from .judge_with_gt import LlmAsJudgeRewardCalculatorWithGT
from .reward import LlmAsJudgeRewardCalculator
from .env_grader import EnvGrader

__all__=[
    "LlmAsJudgeRewardCalculatorWithGT",
    "LlmAsJudgeRewardCalculator",
    "EnvGrader",
    "grader_manager"
]