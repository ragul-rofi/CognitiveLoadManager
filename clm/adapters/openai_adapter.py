"""
OpenAI Agents SDK adapter for CLM.

Usage:
    from clm.adapters import CLMOpenAIHook
    
    hook = CLMOpenAIHook(verbose=True)
    # Pass hook.on_message as a callback to your OpenAI agent runner
    # OR use hook.wrap(runner) to auto-attach
"""
import uuid
from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode


class CLMOpenAIHook:
    """
    Lifecycle hooks for OpenAI Agents SDK.
    
    Compatible with openai-agents SDK on_message_end and on_tool_call hooks.
    
    Usage:
        hook = CLMOpenAIHook(verbose=True)
        
        # Manual hook attachment:
        result = await Runner.run(agent, input,
                                  hooks=hook.get_hooks())
        
        # After run:
        print(hook.clm.summary())
    """
    
    def __init__(self, config: CLMConfig = None, verbose: bool = False):
        self.clm = CognitiveLoadManager(config, verbose=verbose)
        self._history: list[str] = []
        self._tool_nodes: list[str] = []
        session_id = str(uuid.uuid4())[:8]
        root_id = f"root_{session_id}"
        self._task_tree = TaskTree(
            root=TaskNode(root_id, None, "openai_agent_session", "active", depth=0),
            root_intent=""
        )
        self._current_task_id = root_id
    
    def on_message_end(self, message) -> None:
        """Hook: fires after each agent message/LLM response."""
        content = ""
        if hasattr(message, "content"):
            content = str(message.content)
        elif hasattr(message, "text"):
            content = str(message.text)
        
        if not content:
            return
        
        self._history.append(content)
        if not self._task_tree.root_intent:
            self._task_tree.root_intent = content[:300]
        
        state = TaskState(
            task_tree=self._task_tree,
            current_task_id=self._current_task_id,
            reasoning_history=self._history[-3:]
        )
        self.clm.observe(content, state)
    
    def on_tool_call(self, tool_name: str, tool_input: dict) -> None:
        """Hook: fires before each tool call."""
        task_id = f"tool_{tool_name}_{len(self._tool_nodes)}"
        self._tool_nodes.append(task_id)
        node = TaskNode(
            task_id=task_id,
            parent_id=self._task_tree.root.task_id,
            description=f"{tool_name}: {str(tool_input)[:100]}",
            status="active",
            depth=1
        )
        self._task_tree.root.children.append(node)
        self._current_task_id = task_id
    
    def on_tool_end(self, tool_name: str, output: str) -> None:
        """Hook: fires after each tool call."""
        node = self._task_tree.find_node(self._current_task_id)
        if node:
            node.status = "completed"
        self._current_task_id = self._task_tree.root.task_id
    
    def get_hooks(self) -> dict:
        """Return hooks dict compatible with OpenAI Agents SDK Runner."""
        return {
            "on_message_end": self.on_message_end,
            "on_tool_call": self.on_tool_call,
            "on_tool_end": self.on_tool_end,
        }
