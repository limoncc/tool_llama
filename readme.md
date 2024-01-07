#### 一、一个简单介绍

tool_llama是一个配合llama-cpp-python服务使用工具的包. llama-cpp-python目前[V0.2.26]版本, 必须使用如下方式才能使用函数调用功能。 该库使用了[Davor Runje](https://github.com/davorrunje)的对autogenV0.2.3贡献的部分代码，以解决函数输入格式问题。

tool_llama is a package that works with the llama-cpp-python service. llama-cpp-python is currently in [V0.2.26] version and must use the following methods to use function calls.The library USES the [Davor Runje] (https://github.com/davorrunje) to part of the code, the contribution of autogenV0.2.3 function input format, in order to solve problems.

```shell
python3 -m llama_cpp.server --model <model_path> --chat_format functionary
```

这意味着你要使用这个小众的functionary-7b-v1模型, 并在非工具对话上失去chat_format的格式。llama-cpp-python的chat_format=functionary并没有实现其他模型的chat_format。理想情况应该是在通用聊天的情况，也能使用工具。就和openai一样。当前众多开源模型其实在提示词的引导下已经能比较好的使用工具, 所以开发了这个工具，以前完善开源工具链。tool_llama有如下特性：

This means you use the niche functionary-7b-v1 model and lose the chat_format for non-tool conversations. llama-cpp-python's chat_format=functionary does not implement chat_format for other models. Ideally, you should be in a general chat situation where you can also use the tool. Just like openai. At present, many open source models can actually use tools better under the guidance of prompt words, so we developed this tool and improved the open source tool chain before. tool_llama has the following features:

- 1、支持pydantic风格工具输入   
- 2、外部的装饰器风格导入工具与数据模型, 然后使用工具
- 3、支持llama-cpp-python通用聊天模式，不必为了使用工具启动两个模型


- 1, support pydantic style tool input 
- 2, external decorator style import tool with data model, and then use the tool 
- 3, support llama-cpp-python general chat mode, do not need to use the tool to start two models


#### 二、autogen使用范式(autogen mode)

下面是几个例子
Here are a few examples

```python
# 工具
from openai import OpenAI
from pprint import pprint
from tool_llama import Use_Tool
# 类型
from openai import Stream
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from llama_cpp.llama_types import ChatCompletionRequestMessage
from typing import List, Optional, Union

class Expert(BaseModel):
    """专家"""
    
    name: str = Field(description="专家名字, 例如李睿")
    description: str = Field(description="关于专家的技能描述, 尽可能详细")
    ...


class AdvisoryGroup(BaseModel):
    """顾问团"""
    
    name: str = Field(description="顾问团名字")
    target: str = Field(description="顾问团目标任务")
    members: List[Expert] = Field(description="成员")
    ...


toolkit = Use_Tool()

# autogen使用范式
@toolkit.build(tools=[AdvisoryGroup], tool_choice=AdvisoryGroup, namespace="consultative_tool")
def chat_infer(messages: List[ChatCompletionRequestMessage], stream=False, 
               temperature: int = 0.1) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    api_key = "NULL"
    organization = "limoncc"
    base_url = "http://127.0.0.1:8000/v1"
    client = OpenAI(api_key=api_key, organization=organization, base_url=base_url)
    
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=messages,  # type: ignore
        temperature=temperature,
        n=1,
        top_p=1.0,
        presence_penalty=1.1,
        stop=["</s>", '<|im_end|>'],
        max_tokens=3024,
        seed=1010,
        stream=stream
    )
    return response


prompt = "生成一个顾问团队，包含3个专家。以解决AI落地应用为目标。"
msg = [{"role": "user", "content": prompt}, ]
response = chat_infer(msg, stream=True)

result = toolkit.execute_function(response, tools=[AdvisoryGroup])
pprint(result[0], sort_dicts=False)
# output is AdvisoryGroup
# {'name': 'AI落地应用专家团队',
#  'target': '解决AI落地应用',
#  'members': [Expert(name='专家A', description='具备丰富经验的AI应用开发专家'),
#              Expert(name='专家B', description='深入了解企业AI落地实践的行业专家'),
#              Expert(name='专家C', description='拥有多年跨领域AI应用研究经验的学者')]}
```

#### 二、openai使用范式(openai model)

```python
# openai使用范式
toolkit = Use_Tool()
mytools, mytool_choice = toolkit.gen_tools_api_from(tools=[AdvisoryGroup], tool_choice=AdvisoryGroup)

@toolkit.inject(namespace="consultative_tool")
def tool_infer(messages: List[ChatCompletionRequestMessage], stream=False, 
               temperature: int = 0.1, tools: Optional[List] = None, 
               tool_choice: Optional[List] = None) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    api_key = "NULL"
    organization = "limoncc"
    base_url = "http://127.0.0.1:8000/v1"
    client = OpenAI(api_key=api_key, organization=organization, base_url=base_url)
    
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=messages,  # type: ignore
        temperature=temperature,
        n=1,
        top_p=1.0,
        presence_penalty=1.1,
        stop=["</s>", '<|im_end|>'],
        max_tokens=3024,
        seed=1010,
        stream=stream
    )
    return response


prompt = "生成一个顾问团队，包含5个专家。以解决AI落地应用为目标。"
msg = [{"role": "user", "content": prompt}, ]
response = tool_infer(msg, tools=mytools, tool_choice=mytool_choice, stream=False)
result = toolkit.execute_function(response, tools=[AdvisoryGroup])
pprint(result[0], sort_dicts=False)
# output is AdvisoryGroup
# {'name': 'AI落地应用专家团队',
#  'target': '解决AI落地应用',
#  'members': [Expert(name='专家A', description='具备丰富经验的AI应用开发专家'),
#              Expert(name='专家B', description='深入了解企业AI落地实践的行业专家'),
#              Expert(name='专家C', description='拥有多年跨领域AI应用研究经验的学者')]}
```

当然你也可以不使用工具
You can also use no tools.

```python
prompt = "生成一个顾问团队，包含5个专家。以解决AI落地应用为目标。"
msg = [{"role": "user", "content": prompt}, ]
response = tool_infer(msg, stream=True)

for part in response:
    print(part.choices[0].delta.content or "", end='')  # type: ignore

# 建立一个高效的顾问团队需要结合各方面的专业知识和经验。在这里，我们将创建一个五人强大的AI落地应用专家团队：
# 
# 1. 产品顾问 - 艾米莉亚·卢克（Aemilia Luik）
# 艾米莉亚是一位具有丰富经验的产品管理师，擅长将用户需求与技术实现结合起来。她曾成功地为多家企业开发和推出AI应用。
# 2. 数据科学顾问 - 罗伯特·帕尔森（Robert Palmer）
# 罗伯特是一位知名的数据科学家，擅长分析大量数据并提取有价值的见解。他曾为各种行业领域开发过AI算法和模型。
# 3. 人工智能工程师 - 詹姆斯·麦克米尔（James MacMillan）
# 詹姆斯是一位具有深厚技术实力的人工智能工程师，擅长开发和优化AI算法。他曾为多个企业构建了高效、可扩展的AI系统。
# 4. 设计顾问 - 艾米莉亚·帕尔森（Aemilia Palmer）
# 艾米利亚是一位具有丰富经验的产品设计师，擅长创造美观、易用且符合用户需求的界面和交互体验。她曾为多个AI应用程序设计过视觉元素。
# 5. 商业顾问 - 安德鲁·帕尔森（Andrew Palmer）
# 安德鲁是一位经验丰富的企业家，擅长策划和实施成功的营销和推广计划。他曾为多个AI应用程序提供过商业建议，帮助它们在市场上取得成功。
# 
# 这五位专家将共同合作，以最佳方式解决客户面临的各种AI落地问题，从产品设计到实施和推广。他们的多元化技能集会使团队具有强大的力量来帮助企业成功应用人工智能。

```


#### 三、使用自定义prompt

一下是用默认的工具调用提示词,当然你可以修改然后这样使用即可。
The next is to use the default tool to call the prompt word, of course you can modify it and use it as such.

```python
from tool_llama import Use_Tool
toolkit = Use_Tool.from_template("./your_prompt.toml")
```

```toml
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
    
```


