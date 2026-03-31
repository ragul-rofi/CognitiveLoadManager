"""CLM + LangChain — one-line integration example."""
# pip install clm-agent[langchain]

from clm.adapters import CLMCallbackHandler

handler = CLMCallbackHandler(verbose=True)

# Attach to any LangChain agent or chain with callbacks=[handler]
# Example (requires langchain + openai):
#
# from langchain.agents import AgentExecutor
# result = agent_executor.invoke({"input": "your task"}, config={"callbacks": [handler]})
# print(handler.clm.summary())

print("CLMCallbackHandler ready. Attach to any LangChain agent:")
print("  agent.run(input, callbacks=[CLMCallbackHandler(verbose=True)])")
