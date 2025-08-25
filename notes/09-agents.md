# Agents
The recently released large pre-trained models, such as LLMs, created the
opportunity to build agents — intelligent systems that use these models to
plan the execution of complex tasks. Agent workflows are now possible
because of the greater reasoning capabilities of large pre-trained models
such as OpenAI’s latest models.

We can use these models for their deep internal knowledge to create new,
compelling material, think through problems, and make plans. For instance,
we can create a research agent that will find essential facts from different
sources and generate an answer that will combine them in. a helpful way.

## Agents as Intelligent Systems
Agents are systems that use LLMs to determine and order a set of actions.
In a simple workflow, these actions might involve using a tool, examining
its output, and responding to the user’s request. Some essential components
are:
1. Tools: These functions achieve a specific task, such as using the
Google Search API, accessing an SQL database, running code
with a Python REPL, or using a calculator.
2. Reasoning Engine or Core: The large model that powers the
system. The latest large pre-trained models are a great choice
due to their advanced reasoning capabilities.
3. Agent orchestration: The complete system that manages the
interaction between the LLM and its tools.

Agents are generally categorized into two types:
* Action Agents: These agents decide and carry out a single action,
which is good for simple, straightforward tasks.
* Plan-and-Execute Agents: These agents initially develop a plan
with a set of actions and then execute these actions in sequence or
parallel. The results of intermediary actions can also modify the plan
if necessary.

Agents are intelligent systems that use LLMs for reasoning and planning rather than content generation.
