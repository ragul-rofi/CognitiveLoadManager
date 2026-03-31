"""
LangChain adapter for CLM.
Usage:
    from clm.adapters import CLMCallbackHandler
    agent.run(input, callbacks=[CLMCallbackHandler(verbose=True)])
"""
import uuid
from typing import Any

try:
    from langchain.callbacks.base import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LangChain is required for CLMCallbackHandler. "
        "Install it with: pip install clm-agent[langchain]"
    )

from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode


class CLMCallbackHandler(BaseCallbackHandler):
    """
    Drop-in CLM integration for any LangChain agent or chain.
    
    Automatically builds a TaskTree from LLM calls and tool invocations.
    No manual task state construction required.
    
    Usage:
        clm_handler = CLMCallbackHandler(verbose=True)
        agent.run("your task", callbacks=[clm_handler])
        
        # After run, inspect what CLM did:
        print(clm_handler.clm.summary())
    """
    
    def __init__(self, config: CLMConfig = None, verbose: bool = False):
        super().__init__()
        self.clm = CognitiveLoadManager(config, verbose=verbose)
        self._history: list[str] = []
        self._tool_calls: list[str] = []
        self._root_intent: str = ""
        self._session_id = str(uuid.uuid4())[:8]
        
        # Build a live task tree that grows as the agent runs
        root_id = f"root_{self._session_id}"
        self._task_tree = TaskTree(
            root=TaskNode(
                task_id=root_id,
                parent_id=None,
                description="agent_session",
                status="active",
                depth=0
            ),
            root_intent=""
        )
        self._current_task_id = root_id
    
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """Capture root intent from first prompt."""
        if not self._root_intent and prompts:
            self._root_intent = prompts[0][:300]
            self._task_tree.root_intent = self._root_intent
            self._task_tree.root.description = self._root_intent[:100]
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Observe after every LLM call — the core integration point."""
        try:
            output = response.generations[0][0].text
        except (IndexError, AttributeError):
            return
        
        self._history.append(output)
        
        task_state = TaskState(
            task_tree=self._task_tree,
            current_task_id=self._current_task_id,
            reasoning_history=self._history[-3:]
        )
        
        result = self.clm.observe(output, task_state)
        
        # If patch: update task tree with compressed context
        if result.action == "patch" and result.context:
            self._task_tree.root.description = result.context[:200]
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Each tool call is a new sub-task node in the tree."""
        tool_name = serialized.get("name", "unknown_tool")
        task_id = f"tool_{tool_name}_{len(self._tool_calls)}"
        self._tool_calls.append(task_id)
        
        new_node = TaskNode(
            task_id=task_id,
            parent_id=self._task_tree.root.task_id,
            description=f"{tool_name}: {input_str[:100]}",
            status="active",
            depth=1
        )
        self._task_tree.root.children.append(new_node)
        self._current_task_id = task_id
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Mark tool task complete, return focus to root."""
        current = self._task_tree.find_node(self._current_task_id)
        if current:
            current.status = "completed"
        self._current_task_id = self._task_tree.root.task_id
    
    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """Chain finished — print summary if verbose."""
        if self.clm.verbose:
            s = self.clm.summary()
            print(
                f"\n[CLM] Session complete — "
                f"{s['steps']} steps | avg score {s['avg_score']} | "
                f"peak {s['peak_score']} | "
                f"patches={s['interventions']['patch']} "
                f"interrupts={s['interventions']['interrupt']}",
                flush=True
            )
