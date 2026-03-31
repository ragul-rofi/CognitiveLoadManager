"""
Generic loop adapter for CLM. Works with any LLM: OpenAI, Anthropic, Gemini, local models.

Usage:
    from clm.adapters import CLMLoop
    
    @CLMLoop(verbose=True)
    def my_agent_step(prompt: str) -> str:
        return openai_client.chat(prompt)  # or any LLM call
    
    # Now call it normally — CLM wraps every step automatically
    for i in range(10):
        response = my_agent_step(current_prompt)
"""
import uuid
import functools
from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode


class CLMLoop:
    """
    Decorator that wraps any agent step function with CLM.
    
    The decorated function must accept a string prompt and return a string output.
    CLM observes every call automatically, maintaining its own internal task tree.
    
    Usage as decorator:
        @CLMLoop(verbose=True)
        def agent_step(prompt: str) -> str:
            return llm_call(prompt)
    
    Usage as context manager:
        with CLMLoop(verbose=True) as loop:
            for step in range(max_steps):
                output = loop.step(prompt, task_description="current subtask")
                if loop.should_stop():
                    break
    
    Access CLM internals any time:
        loop.clm.summary()
        loop.clm.get_history()
        loop.clm.get_score()
    """
    
    def __init__(self, config: CLMConfig = None, verbose: bool = False):
        self.clm = CognitiveLoadManager(config, verbose=verbose)
        self._history: list[str] = []
        self._session_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._root_intent = ""
        
        root_id = f"root_{self._session_id}"
        self._task_tree = TaskTree(
            root=TaskNode(root_id, None, "loop_root", "active", depth=0),
            root_intent=""
        )
        self._current_task_id = root_id
    
    def __call__(self, fn):
        """Decorator usage: @CLMLoop()"""
        @functools.wraps(fn)
        def wrapper(prompt: str, *args, **kwargs) -> str:
            output = fn(prompt, *args, **kwargs)
            self._observe(prompt, output)
            return output
        wrapper.clm = self.clm
        wrapper.summary = self.clm.summary
        wrapper.get_history = self.clm.get_history
        return wrapper
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.clm.close()
    
    def step(self, prompt: str, output: str = None,
             task_description: str = None) -> dict:
        """
        Manually advance one step. Pass prompt + output, get CLM result back.
        If output is None, returns a partial state (pre-LLM call marker).
        
        Returns the InterventionResponse as a dict so callers can branch on it.
        """
        if output is None:
            # pre-call: just record the intent
            if not self._root_intent:
                self._root_intent = prompt[:300]
                self._task_tree.root_intent = self._root_intent
            return {"action": "pending"}
        
        result = self._observe(prompt, output, task_description)
        return {
            "action": result.action,
            "zone": result.zone,
            "score": result.clm_score,
            "context": result.context,
            "clarification": result.clarification,
        }
    
    def should_stop(self) -> bool:
        """
        Returns True if CLM is in Red zone — useful for agents that want to
        pause and ask for clarification rather than continuing blind.
        """
        return self.clm.get_zone() == "Red"
    
    def _observe(self, prompt: str, output: str,
                 task_description: str = None) -> object:
        self._step_count += 1
        self._history.append(output)
        
        if not self._root_intent:
            self._root_intent = prompt[:300]
            self._task_tree.root_intent = self._root_intent
        
        # Add step as a child node
        step_id = f"step_{self._step_count}"
        node = TaskNode(
            task_id=step_id,
            parent_id=self._task_tree.root.task_id,
            description=task_description or output[:100],
            status="active",
            depth=1
        )
        self._task_tree.root.children.append(node)
        self._current_task_id = step_id
        
        state = TaskState(
            task_tree=self._task_tree,
            current_task_id=self._current_task_id,
            reasoning_history=self._history[-3:]
        )
        return self.clm.observe(output, state)
