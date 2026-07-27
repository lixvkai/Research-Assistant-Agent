"""全项目共享的 Pydantic 数据模型。

集中定义三类契约：
1. 任务规划（Plan / SubTask / ExpertName）—— 服务于 Orchestrator
2. ReAct 事件（StepStart / Thought / ... / ReActEvent）—— 服务于 react_agent ↔ UI
3. 工具规范（ToolSpec / ToolInfo）—— 服务于 MCP 工具层

设计约束：ReAct 事件的 `type` 字符串值与历史实现保持完全一致，
以保证 Gradio UI 的流式渲染零回归。
"""

from enum import Enum
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ── 1. 任务规划 ────────────────────────────────────────────────────

class ExpertName(str, Enum):
    """合法的专家名枚举，避免裸字符串拼写漂移。"""
    literature = "literature"
    data_analysis = "data_analysis"
    writing = "writing"
    review = "review"


class SubTask(BaseModel):
    expert: ExpertName
    task: str
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    plan_summary: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)


# ── 2. ReAct 事件 ──────────────────────────────────────────────────

class StepStart(BaseModel):
    type: Literal["step_start"] = "step_start"
    step: int
    max_steps: int


class Thought(BaseModel):
    type: Literal["thought"] = "thought"
    content: str
    step: int


class Action(BaseModel):
    type: Literal["action"] = "action"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    step: int


class Observation(BaseModel):
    type: Literal["observation"] = "observation"
    result: str
    step: int


class Answer(BaseModel):
    type: Literal["answer"] = "answer"
    content: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    content: str


class ReflectionEvent(BaseModel):
    type: Literal["reflection"] = "reflection"
    sufficient: bool
    critique: str
    step: int = 0


ReActEvent = Union[
    StepStart, Thought, Action, Observation, Answer,
    ReflectionEvent, ErrorEvent,
]


# ── 反思判定（结构化输出） ─────────────────────────────────────────

class Reflection(BaseModel):
    """自我反思的判定结果。解析失败时默认 sufficient=True，避免死循环。"""
    sufficient: bool = True
    critique: str = ""


# ── 3. 工具规范 ────────────────────────────────────────────────────

class ToolSpec(BaseModel):
    """MCP 工具规范（替代原 dataclass）。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable
    category: str = "general"
    version: str = "1.0"


class ToolInfo(BaseModel):
    """对外暴露的工具元信息（供 UI 工具箱渲染）。"""
    name: str
    description: str
    category: str
    version: str
