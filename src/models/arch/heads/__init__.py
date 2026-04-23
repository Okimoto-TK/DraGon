from .dual_domain_concat_head import DualDomainConcatHead
from .multi_task_heads import MultiTaskHeads
from .single_task_head import SingleTaskHead
from .task_query_tower import TaskQueryTower

__all__ = ["DualDomainConcatHead", "TaskQueryTower", "MultiTaskHeads", "SingleTaskHead"]
