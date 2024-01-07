# -*- coding: utf-8 -*-
# ==============================
# 作者: 李健/limoncc
# 邮箱: limoncc@icloud.com
# 日期：2024/1/5 17:02
# 标题：解决开源模型使用工具问题
# 内容：把工具转为系统提示词
# ==============================

import json
import rtoml
from tool_llama.utils.function_utils import get_function_schema
from tool_llama.utils.basemodel_utils import get_model_schema

# 引入类型
from pathlib import Path
from inspect import isfunction
from tool_llama.utils.basemodel_utils import BaseModel
from typing import Dict, Union, Type, Callable, Any, List, Tuple, Protocol, Optional, Iterator

# llama相关类型

from llama_cpp.llama_types import JsonType
from llama_cpp.llama_types import ChatCompletionRequestMessage
from llama_cpp.llama_types import ChatCompletionRequestSystemMessage
# from llama_cpp.llama_types import CreateChatCompletionResponse
# from llama_cpp.llama_types import CreateChatCompletionStreamResponse
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

Function = Callable[..., Any]
Tool = Union[Type[BaseModel], Tuple[Function, str]]


class LLM_INFER(Protocol):
    def __call__(self, messages: List[ChatCompletionRequestMessage]
                 , stream: bool
                 , **kwargs
                 ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        ...


class Use_Tool:
    guide_prompt = """You are a helpful AI Assistant. You polite answers to the user's questions.
1. If the question is unclear, Please polite ask the user about context.
2. if the task has no available function in the namespace, Please reply:
    There is no function to call to complete the task, here is my plan for completing the task:
    [you plan steps]
3. if you calls functions with appropriate input when necessary. Please strictly reply the following format json to the question. Don't have extra characters.
{
    "functions": //不要省略
    [{  "namespace": namespace_name,
        "name": function_name,
        "arguments": **kwargs
    }]
}//end
3. For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.

"""
    tools_prompt_prefix = "// Supported function definitions that should be called when necessary."
    tools_prompt_suffix = "note: 请使用中文回答。"
    
    def __init__(self, guide_prompt: Optional[str] = None, tools_prompt_prefix: Optional[str] = None, tools_prompt_suffix: Optional[str] = None):
        if guide_prompt:
            self.guide_prompt = guide_prompt
        else:
            self.guide_prompt = self.guide_prompt
        if tools_prompt_prefix:
            self.tools_prompt_prefix = tools_prompt_prefix
        else:
            self.tools_prompt_prefix = self.tools_prompt_prefix
        if tools_prompt_suffix:
            self.tools_prompt_suffix = self.tools_prompt_suffix
        else:
            self.tools_prompt_suffix = self.tools_prompt_suffix
        ...
    
    @classmethod
    def from_template(cls, tools_prompt_path: Path):
        with tools_prompt_path.open("r") as f:
            tools_prompt = rtoml.load(f)
        return cls(tools_prompt['guide_prompt'], tools_prompt['tools_prompt_prefix'], tools_prompt['tools_prompt_suffix'])
    
    @classmethod
    def check_tools(cls, tools: Union[Tool, List[Tool]]) -> str:
        if isinstance(tools, tuple):
            if isfunction(tools[0]) and isinstance(tools[1], str):
                return "Function"
            else:
                raise TypeError("Type of tools is errors.")
        elif isinstance(tools, list):
            return "List"
        else:
            try:
                if tools.__base__ == BaseModel:
                    return "BaseModel"
            except Exception as e:
                raise TypeError("Type of tools is errors.")
        ...
    
    @classmethod
    def gen_type_definition(cls, param: Dict[str, JsonType], indent_level: int, shared_defs) -> str:
        indent = "  " * indent_level
        if "$ref" in param:
            # Reference to a shared definition
            ref_name = param["$ref"].split("/")[
                -1
            ]  # Extract the type name from the reference
            return ref_name
        elif param.get("type") == "array":
            items = param.get("items", {})
            item_type = cls.gen_type_definition(items, indent_level + 1, shared_defs)
            return f"Array<{item_type}>"
        elif param.get("type") == "object":
            properties = param.get("properties", {})
            nested_schema = "{\n"
            for nested_param_name, nested_param in properties.items():
                nested_param_type = cls.gen_type_definition(
                    nested_param, indent_level + 1, shared_defs
                )
                nested_schema += (
                    f"{indent}  {nested_param_name}: {nested_param_type},\n"
                )
            nested_schema += indent + "}"
            return nested_schema
        elif "enum" in param:
            # Enum type
            return " | ".join([f'"{enum_value}"' for enum_value in param["enum"]])
        # 这里添加新的类型
        elif param.get("type") == "integer":
            return "number"
        else:
            # Simple type
            return param.get("type", "any")
    
    @classmethod
    def gen_shared_definitions(cls, shared_defs, indent_level: int) -> str:
        indent = "  " * indent_level
        shared_definitions = ""
        for def_name, def_properties in shared_defs.items():
            shared_definitions += f"{indent}type {def_name} = "
            if def_properties.get("type") == "object":
                shared_definitions += cls.gen_type_definition(
                    def_properties, indent_level, shared_defs
                )
            elif "enum" in def_properties:
                # Enum type
                shared_definitions += " | ".join(
                    [f'"{enum_value}"' for enum_value in def_properties["enum"]]
                )
            shared_definitions += ";\n"
        return shared_definitions
    
    @classmethod
    def gen_schema_from_functions(cls, functions: List, namespace="functions") -> str:
        schema = ""
        # schema = (
        #     "// Supported function definitions that should be called when necessary.\n"
        # )
        schema += f"namespace {namespace} {{\n\n"
        
        # Generate shared definitions
        shared_definitions = {}
        for function in functions:
            parameters = function.get("parameters", {})
            shared_definitions.update(parameters.get("$defs", {}))
        
        schema += cls.gen_shared_definitions(shared_definitions, 1)
        
        for function in functions:
            function_name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            required_params = parameters.get("required", [])
            
            schema += f"  // {description}\n"
            schema += f"  type {function_name} = (_: {{\n"
            
            for param_name, param in parameters.get("properties", {}).items():
                param_description = param.get("description", "")
                param_type = cls.gen_type_definition(param, 2, shared_definitions)
                optional_indicator = "" if param_name in required_params else "?"
                schema += f"    // {param_description}\n"
                schema += f"    {param_name}{optional_indicator}: {param_type},\n"
            schema += "  }) => any;\n\n"
        
        schema += "}} // namespace {}\n".format(namespace)
        return schema
    
    @classmethod
    def gen_schema_from_tools(cls, tools: Union[Tool, List[Tool]], namespace="functions") -> str:
        
        functions, _ = cls.gen_tools_api_from(tools)
        schema = cls.gen_schema_from_functions(functions, namespace)
        return schema
    
    def message_inject_tools(self, tools: Union[Tool, List[Tool]], namespace="functions") -> List[ChatCompletionRequestSystemMessage]:
        tools_prompt = self.gen_schema_from_tools(tools, namespace=namespace)
        system_guide = {"role": "system", "content": f"{self.guide_prompt}\n{self.tools_prompt_prefix}\n{tools_prompt}\n{self.tools_prompt_suffix}"}
        # system_tools = {"role": "system", "content": f""}
        system_messages = [ChatCompletionRequestSystemMessage(**system_guide)]
        
        return system_messages
    
    @classmethod
    def get_tool_name(cls, tools: Union[Tool, List[Tool]]) -> List[str]:
        tools_name = []
        if cls.check_tools(tools) == 'BaseModel':
            tools_name.append(tools.model_json_schema()['title'])
        elif cls.check_tools(tools) == 'Function':
            tools_name.append(tools[0].__name__)
        elif cls.check_tools(tools) == 'List':
            for tool in tools:
                if cls.check_tools(tool) == 'Function':
                    tools_name.append(tool[0].__name__)
                elif cls.check_tools(tool) == "BaseModel":
                    tools_name.append(tool.model_json_schema()['title'])
                else:
                    raise TypeError("Type of tools is errors.")
                ...
            ...
        else:
            raise TypeError("Type of tools is errors.")
        
        return tools_name
    
    @classmethod
    def format_json_str(cls, json_str: str):
        """Remove newlines outside of quotes, and handle JSON escape sequences.

        1. this function removes the newline in the query outside of quotes otherwise json.loads(s) will fail.
            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"

        2. this function also handles JSON escape sequences inside quotes,
            Ex 1:
            '{"args": "a\na\na\ta"}' -> '{"args": "a\\na\\na\\ta"}'
        """
        result = []
        inside_quotes = False
        last_char = " "
        for char in json_str:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)
    
    @classmethod
    def deal_text(cls, text: str) -> List:
        
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index != -1 and end_index != -1:
            json_str = text[start_index:end_index + 1]
            if "functions" in json_str:
                json_str_formated = Use_Tool.format_json_str(json_str)
                functions_call = json.loads(json_str_formated)
                functions = functions_call['functions']
                # functions_name = [item['name'] for item in functions_call['functions']]
                # functions_arguments = [item['arguments'] for item in functions_call['functions']]
                return functions
            else:
                raise ValueError("无法解析JSON数据，原因是: 没有functions")
    
    @classmethod
    def gen_tool_response(cls, stream: bool, response: Union[ChatCompletion, Stream[ChatCompletionChunk]]):
        chat_id = ""
        created = ""
        model = ""
        usage = ""
        usage = ""
        if stream:
            try:
                context = ""
                for part in response:
                    chat_id = part.id
                    created = part.created
                    model = part.model
                    # usage = part.usage
                    usage = {}
                    context += part.choices[0].delta.content or ""
                functions = cls.deal_text(context)
            except Exception as errors:
                raise ValueError("无法解析JSON数据，原因是：", str(errors))
        else:
            try:
                chat_id = response.id
                created = response.created
                model = response.model
                usage = response.usage
                
                context = response.choices[0].message.content
                functions = cls.deal_text(context)
            except Exception as errors:
                raise ValueError("无法解析JSON数据，原因是：", str(errors))
        try:
            new_response = ChatCompletion(
                id=chat_id,
                object="chat.completion",
                created=created,
                model=model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": json.dumps([item['name'] for item in functions], ensure_ascii=False),
                                "arguments": json.dumps([item['arguments'] for item in functions], ensure_ascii=False),
                            },
                            "tool_calls": [
                                {
                                    "id": item['name'],
                                    "type": "function",
                                    "function": {
                                        "name": item['name'],
                                        "arguments": json.dumps(item['arguments'], ensure_ascii=False),
                                    },
                                } for item in functions],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                usage=usage
            )
        except Exception as errors:
            raise ValueError("无法解析JSON数据，原因是：", str(errors))
        return new_response
    
    def build(self, tools: Optional[Union[Tool, List[Tool]]] = None, tool_choice: Optional[Union[Tool, List[Tool]]] = None, namespace="functions"):
        def decorator(func: LLM_INFER):
            def wrapper(*args, **kwargs):
                if tools:
                    if tool_choice:
                        tools_name = self.get_tool_name(tool_choice)
                        # 强制使用工具不太好在API外部解决。代开发
                        ...
                    # 强制使用stream，未来可能支持stream
                    kwargs["stream"] = False
                    system_messages = self.message_inject_tools(tools, namespace)
                    # 注入工具
                    if len(kwargs.get('messages', [])) == 0:
                        # 更新args[0]
                        messages = args[0]
                        args = (system_messages + messages,)
                    else:
                        # 更新这个
                        messages = kwargs['messages']
                        kwargs['messages'] = [*system_messages, *messages]
                    # 这个应该怎么实现
                response = func(*args, **kwargs)
                if tools:
                    if "namespace" in response.choices[0].message.content:
                        response = self.gen_tool_response(kwargs['stream'], response)
                return response
            
            return wrapper
        
        return decorator
    
    @classmethod
    def get_tool_dict(cls, tools: Union[Tool, List[Tool]]) -> Dict[str, Callable]:
        tool_dict = {}
        if cls.check_tools(tools) == 'BaseModel':
            tool_dict[tools.model_json_schema()['title']] = tools
        elif cls.check_tools(tools) == 'Function':
            tool_dict[tools[0].__name__] = tools[0]
        elif cls.check_tools(tools) == 'List':
            for tool in tools:
                if cls.check_tools(tool) == 'Function':
                    tool_dict[tool[0].__name__] = tool[0]
                elif cls.check_tools(tool) == "BaseModel":
                    tool_dict[tool.model_json_schema()['title']] = tool
                else:
                    raise TypeError("Type of tools is errors.")
                ...
            ...
        else:
            raise TypeError("Type of tools is errors.")
        
        return tool_dict
    
    @classmethod
    def execute_function(cls, response: Union[ChatCompletion, Stream[ChatCompletionChunk]], tools: Union[Tool, List[Tool]]) -> List[Any]:
        result = []
        tool_dict = cls.get_tool_dict(tools)
        if isinstance(response, Stream):
            pass
        elif isinstance(response, ChatCompletion):
            if response.choices[0].message.tool_calls:
                for function in response.choices[0].message.tool_calls:
                    function_name = function.function.name
                    function_arguments = json.loads(function.function.arguments)
                    result.append(tool_dict[function_name](**function_arguments))
        else:
            raise TypeError("the response is TypeError.")
        return result
    
    @classmethod
    def gen_tools_api_from(cls, tools: Union[Tool, List[Tool]], tool_choice: Optional[Union[Tool, List[Tool]]] = None) -> Tuple[List, List]:
        functions = []
        functions_choice = []
        if cls.check_tools(tools) == "Function":
            functions.append(get_function_schema(tools[0], description=tools[1]))
        elif cls.check_tools(tools) == "BaseModel":
            functions.append(get_model_schema(tools))
        elif cls.check_tools(tools) == "List":
            for tool in tools:
                if cls.check_tools(tool) == "Function":
                    functions.append(get_function_schema(tool[0], description=tool[1]))
                elif cls.check_tools(tool) == "BaseModel":
                    functions.append(get_model_schema(tool))
                else:
                    raise TypeError("Type of tools is errors.")
                ...
            ...
        else:
            raise TypeError("Type of tools is errors.")
        ...
        if tool_choice:
            functions_choice = cls.get_tool_name(tool_choice)
        return functions, functions_choice
        ...
    
    def inject(self, namespace="functions"):
        def decorator(func: LLM_INFER):
            def wrapper(*args, **kwargs):
                is_tools = "tools" in kwargs.keys()
                if kwargs:
                    if is_tools:
                        tools = kwargs['tools']
                        tools_prompt = self.gen_schema_from_functions(tools, namespace=namespace)
                        system_guide = {"role": "system", "content": f"{self.guide_prompt}\n{self.tools_prompt_prefix}\n{tools_prompt}\n{self.tools_prompt_suffix}"}
                        system_messages = [ChatCompletionRequestSystemMessage(**system_guide)]
                        _ = kwargs.pop('tools')
                        if len(kwargs.get('messages', [])) == 0:
                            # 更新args[0]
                            messages = args[0]
                            args = (system_messages + messages,)
                        else:
                            # 更新这个
                            messages = kwargs['messages']
                            kwargs['messages'] = [*system_messages, *messages]
                    if "tools" in kwargs.keys() and "tool_choice" in kwargs.keys():
                        # 这个应该怎么实现
                        _ = kwargs.pop('tool_choice')
                        ...
                    ...
                if is_tools:
                    kwargs["stream"] = False
                    stream = False
                    response = func(*args, **kwargs)
                    if "namespace" in response.choices[0].message.content:
                        response = self.gen_tool_response(stream, response)
                else:
                    response = func(*args, **kwargs)
                ...
                return response
            
            return wrapper
        
        return decorator
