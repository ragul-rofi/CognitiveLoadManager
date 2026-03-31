"""
Auto-builds TaskState from raw agent outputs — no manual tree construction.

Heuristics used:
- Each LLM output becomes a leaf node in the task tree
- Numbered steps, bullet points, and "Step N:" patterns become sub-tasks
- Root intent is inferred from the first observed output
- Branching is inferred from parallel action markers ("also", "meanwhile", "and also")
"""
import re
import uuid
from clm.core.models import TaskState, TaskTree, TaskNode


class AutoStateBuilder:
    """
    Incrementally builds a TaskState from raw LLM outputs.
    No manual task tree required — just feed it outputs.
    
    Usage:
        builder = AutoStateBuilder()
        builder.observe(llm_output_1)
        builder.observe(llm_output_2)
        task_state = builder.get_state()  # ready for clm.observe()
    """
    
    def __init__(self):
        self._session_id = str(uuid.uuid4())[:8]
        self._root_id = f"root_{self._session_id}"
        self._step = 0
        self._history: list[str] = []
        self._nodes: list[TaskNode] = []
        self._root_intent = ""
        
        self._root = TaskNode(
            task_id=self._root_id,
            parent_id=None,
            description="agent_session",
            status="active",
            depth=0
        )
        self._tree = TaskTree(
            root=self._root,
            root_intent=""
        )
    
    def observe(self, llm_output: str) -> None:
        """Feed a new LLM output to update the internal task tree."""
        self._step += 1
        self._history.append(llm_output)
        
        if not self._root_intent:
            self._root_intent = llm_output[:300]
            self._tree.root_intent = self._root_intent
            self._root.description = llm_output[:100]
        
        # Detect sub-tasks from output structure
        sub_tasks = self._extract_subtasks(llm_output)
        
        if sub_tasks:
            for i, task_desc in enumerate(sub_tasks):
                node = TaskNode(
                    task_id=f"step_{self._step}_sub_{i}",
                    parent_id=self._root_id,
                    description=task_desc,
                    status="active",
                    depth=1
                )
                self._root.children.append(node)
                self._nodes.append(node)
        else:
            node = TaskNode(
                task_id=f"step_{self._step}",
                parent_id=self._root_id,
                description=llm_output[:150],
                status="active",
                depth=1
            )
            self._root.children.append(node)
            self._nodes.append(node)
    
    def get_state(self) -> TaskState:
        """Return the current TaskState — pass directly to clm.observe()."""
        current_id = (
            self._nodes[-1].task_id if self._nodes else self._root_id
        )
        return TaskState(
            task_tree=self._tree,
            current_task_id=current_id,
            reasoning_history=self._history[-3:]
        )
    
    def reset(self) -> None:
        """Start a fresh session — call between agent runs."""
        self.__init__()
    
    def _extract_subtasks(self, text: str) -> list[str]:
        """
        Heuristically extract sub-tasks from LLM output.
        Detects numbered lists, bullet points, and step markers.
        """
        patterns = [
            r"^\s*\d+[\.\)]\s+(.+)$",          # 1. task or 1) task
            r"^\s*[-*•]\s+(.+)$",               # - task or * task
            r"(?i)step\s+\d+[:\s]+(.+)$",       # Step 1: task
            r"(?i)(?:first|then|next|finally)[,:\s]+(.+?)(?:\.|$)",  # first, do X
        ]
        tasks = []
        for line in text.split("\n"):
            for pattern in patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    task_text = match.group(1).strip()
                    if len(task_text) > 5:  # Skip trivial matches
                        tasks.append(task_text[:150])
                    break
        return tasks[:10]  # Cap at 10 sub-tasks per output
