from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from code import InteractiveConsole
from langchain.llms import OpenAI
import json
import os
from typing import List, Callable, Dict

class EmptyCallbackHandler(BaseCallbackHandler):
    pass

def save_to_file(goal: str, result: dict):
    """Saves the results dict (second argument) to a file ending in '.result.txt' with the name set to the goal (first argument)"""
    fn = goal.replace(' ','_') + '.result.txt'
    print(f'saving final result to {fn}')
    with open(fn, 'w') as f:
        json.dump(result, f)

def convert_langchain_tools(tools: List[BaseTool]) -> List[dict]:
    """Converts a list of BaseTools (used in langchain) to a list of dictionaries containing the keys: 'name', and 'description'."""
    return [{'name': tool.name, 'description': tool.description} for tool in tools if isinstance(tool, BaseTool)]

class TaskManager(object):
    """Task Manager"""
    current_tasks: List[str] = []
    final_goal: str
    goal_completed: bool = False
    tools: List[dict]
    verbose: bool = True
    llm: BaseLLM
    final_result: dict = {}
    stored_info: dict = {}
    persist: str = None
    completed_tasks: dict = {}
    BASE_PROMPT: str = """
    You are a task management system. Your job is to create, reformulate, and refine a set of tasks. The tasks must be focused on achieving your final goal. It is very important you keep the final goal in mind as you think. Your goal is a constant, throughout, and will never change. 
    As tasks are completed, update your stored info with any info you will need to output at the end. As you go, add on to your final result. Your final result will be returned once, either, you cannot come up with any more reasonable tasks and all are complete, or your final result satisfies your final goal. 
    The language models assigned to your tasks will have access to a list of tools available. As language models, you cannot interact with the internet, however the following tools have been made available so that the final goal can be met. As the tasks you create will be given to other agents, make sure to be specific with each tasks instructions.

    Tools
    -----
    {tools_str}
    -----

    Final Goal
    ----------
    {final_goal}
    ----------

    Current values
    --------------
    current_tasks: {current_tasks}
    stored_info: {stored_info}
    final_result: {final_result}
    --------------
    """
    ENSURE_COMPLETE_PROMPT: str = '''
    Based on your current values, assess whether you have completed your final_goal. Respond with a dictionary in valid JSON format with the following keys:
    "final_result" - dict - reformat your final result to better meet your final goal,
    "goal_complete" - bool - True if you have completed your final_goal, otherwise False if you need to continue
    "current_tasks" - list - list of strings containing tasks in natural language which you will need to complete to meet your final_goal. leave this empty if you set "goal_complete" to True.

    You always give your responses in VALID, JSON READABLE FORMAT.
    '''
    REFINE_PROMPT: str = '''
    Task Result
    -----------
    task: {task}
    result: {result}
    -----------
    
    Refine your current set of tasks based on the task result above. E.g., if information has already been gathered that satisfies the requests in a task, it is not needed anymore. However, if information gathered shows a new task needs to be added, include it.
    If the result included any info you may need to complete later tasks, add it to your stored_info.
    If the result included any info you may need to satisfy your final goal, add it to the final result. Format it as necessary, but make sure it includes all information needed.
    You always give your response in valid JSON format so that it can be parsed (in python) with `json.loads`. Return a dictionary with the keys: "current_tasks" a list of strings (your complete set of tasks, if you need to, add any new tasks and reorder as you see fit), "final_result" a dict (your final result to satisfy your final goal, add to this as you go), "stored_info" a dict (info you may need for later tasks), if you have any thoughts to output to the user, include them as a string with the key "thoughts", and lastly, the key "goal_complete" should contain a boolean value True or False indicating if the final goal has been reached. 
    Make sure your list of tasks ends with a final task like "show results and terminate".
    '''
    TASK_PROMPT = '''
    You are one of many language models working on the same final goal: {final_goal}.

    Here is the list of tasks after yours needed to achieve this: {current_tasks}. Your job is to complete this one task: {task}.

    Here is some context from previous task results: {combined_info}. 

    {task}
    '''
    CREATE_PROMPT: str = 'Based on your end goal, come up with a list of tasks (in order) that you will need to take to achieve your goal.\nGive your response in valid JSON format so that it can be parsed (in python) with `json.loads`. Return a dictionary with the key "current_tasks" containing a list of strings. Make sure your list of tasks ends with a final step such as "show results and terminate".'
    FIX_JSON_PROMPT: str = """
    Reformat the following JSON without losing content so that it can be loaded without errors in python using `json.loads`. The following output returned an error when trying to parse. Make sure your response doesn't contain things like: new lines, tabs. Make sure your response uses double quotes as according to the JSON spec. Your response must include an ending quote and ending bracket as needed. ONLY RETURN VALID JSON WITHOUT FORMATTING. 

    Example of valid JSON: {example}

    Bad JSON: {bad_json}

    Error: {err}

    Good JSON: """
    GOOD_JSON_EXAMPLE: str = '''{"current_tasks": ["Research Amjad Masad's career and background.", "Create a CSV called \"career.csv\" and write his careers to it."], "stored_info": {"username": "amasad"}, "thoughts": "I will research his career and background, and then save the results to \"career.csv\"."}'''

    def _make_tools_str(self, tools: List[dict]) -> str:
        """Tools should be a list of dictionaries with the keys: "name" and "description"."""
        return '-----\n'.join(['\n'.join([f'{k}: {v}' for k, v in tool.items()]) for tool in tools])

    def _load_persist(self):
        if os.path.exists(self.persist):
            with open(self.persist, 'r') as f:
                saved = json.load(f)
            self.output_func(f'[system] Loaded stored info from: {self.persist}')
        else:
            saved = {}
            self.output_func(f'[system] Could not read {self.persist}, assuming new file. It will be created later.')
        self.stored_info = saved.get('stored_info', {})
        self.final_result = saved.get('final_result', {})
        self.current_tasks = saved.get('current_tasks', [])
        self.completed_tasks = saved.get('completed_tasks', {})

    def _save_persist(self):
        with open(self.persist, 'w') as f:
            json.dump({
                'stored_info': self.stored_info,
                'final_result': self.final_result,
                'current_tasks': self.current_tasks,
                'completed_tasks': self.completed_tasks
            }, f)
        self.output_func(f'saved stored info to: {self.persist}')
    
    def __init__(self, goal: str, tools: List[dict], llm: BaseLLM, verbose: bool = True, output_func: Callable = print, complete_func: Callable = save_to_file, input_func: Callable = input, current_tasks: List[str] = None, final_result: dict = None, allow_repeat_tasks: bool = True, completed_tasks: dict = None, persist: str = None, confirm_tool: bool = False):
        """
        :param goal: str - final goal in natural language
        :param tools: List[dict] - a list of tools (dicts) containing keys "name" and "description"
        :param llm: BaseLLM - language model to be used
        :param verbose: bool - whether to output status messages
        :param output_func: Callable - function used to output text (default: print)
        :param complete_func: Callable - function to call when goal is complete
        :param input_func: Callable - function to call for user input (default: input)
        :param current_tasks: List[str] - list of current tasks (optional, used to persist state)
        :param final_result: dict - final result dict (optional, used to persist state)
        :param allow_repeat_tasks: bool - whether to allow tasks to be repeated (default: True)
        :param completed_tasks: dict - dictionary of completed tasks (optional, used to persist state)
        :param persist: str - path to persist data (optional)
        :param confirm_tool: bool - whether to confirm the tool usage
        """
        self.final_goal = goal
        self.tools = tools
        self.verbose = verbose
        self.llm = llm
        self.output_func = output_func
        self.complete_func = complete_func
        self.input_func = input_func
        self.allow_repeat_tasks = allow_repeat_tasks
        self.completed_tasks = completed_tasks or {}
        self.persist = persist
        self.confirm_tool = confirm_tool
        self.current_tasks = current_tasks or []
        self.final_result = final_result or {}
        self.stored_info = {}

        if persist:
            self._load_persist()
        self.tools_str = self._make_tools_str(self.tools)
