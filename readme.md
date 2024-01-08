English | [简体中文](readme_cn.md)

#### One. A Simply Introduce

The tool_llama is an aid package specifically designed for seamless integration with the llama-cpp-python service. The current version of this service stands at [V0.2.26] make use of its function tool, it’s crucial to adhere to a particular usage approach as follows:

```shell
python3 -m llama_cpp.server --model <model_path> --chat_format functionary
```

This implies employing the less renowned functionary-7b-v1 model and sacrificing chat_format in non-tool discussions. In llama-cpp-python, the chat_format is set as ‘functionary’, which has’t achieved compatibility with other models’ chat_formats yet. Ideally, we should be able to use tools within general chats as well, similar to OpenAI’s approach. Most existing open-source models can now leverage tools effectively when guided by prompts. Consequently, tool_llama was developed to enhance the capabilities of current open-source toolchains. It boasts the following notable features:


- 1, Offers compatibility with Pydantic style tool input formatting
- 2, Externally, it facilitates an import of tools and data models in a decorator fashion before their utilization
- 3, Enables the use of tools within llama-cpp-python’s general chat mode without necessitating the startup of two separate models

you can install by pip:

```shell
pip install tool_llama
```


#### Two. autogen mode

Starting a model server by llama-cpp-python. NeuralHermes-2.5-Mistral-7B is recommended by me. 

```shell
python -m llama_cpp.server \
--model ../models/mlabonne/NeuralHermes-2.5-Mistral-7B/Q4_K_M.gguf \
--n_gpu_layers -1 --n_ctx 4096 --chat_format chatml
```

Here are an examples with using BaseModel.

```python
# tools
from openai import OpenAI
from pprint import pprint
from tool_llama import Use_Tool

# types
from openai import Stream
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from llama_cpp.llama_types import ChatCompletionRequestMessage
from typing import List, Union, Annotated


class Expert(BaseModel):
    """Expert"""
    
    name: str = Field(description="Expert name, such as Li Rui")
    description: str = Field(description="Describe the expert's skills in as much detail as possible")
    ...


class AdvisoryGroup(BaseModel):
    """Advisory Group"""
    
    name: str = Field(description="Name of the advisory group")
    target: str = Field(description="Advisory board objective mission")
    members: List[Expert] = Field(description="members")
    ...


toolkit = Use_Tool()


# autogen model
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


prompt = "Generate a team of advisors, consisting of 3 experts. To solve the AI landing application as the goal."
msg = [{"role": "user", "content": prompt}, ]
response = chat_infer(msg, stream=True)

result = toolkit.execute_function(response, tools=[AdvisoryGroup])
pprint(result[0].__dict__, sort_dicts=False)
```

the result you will get:
```shell
{'name': 'AI Landing Application Advisory Group',
 'target': 'Solving AI Landing Application Goal',
 'members': [Expert(name='Expert 1', description='An expert in AI and landing page optimization'),
             Expert(name='Expert 2', description='A professional with expertise in application development and user experience design'),
             Expert(name='Expert 3', description='An expert in AI-based problem solving and data analysis for landing page optimization')]}
```

For the python function, You need to do this as follows:

```python
from typing import List, Union, Annotated
from tool_llama import Use_Tool

def get_weather_from(location: Annotated[str, "地点"]) -> str:
    return "The sun warm and bright."

tools = [(get_weather_from, "Get the weather at the location")]
toolkit = Use_Tool()

@toolkit.build(tools=tools, namespace="functions")
def chat_infer(messages, stream=False,):
    ...
```


#### Three. openai model

```python
from tool_llama import Use_Tool

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
```

Of course, you can also go without tools:

```python
prompt = "Generate a team of advisors, consisting of 3 experts. To solve the AI landing application as the goal."
msg = [{"role": "user", "content": prompt}, ]
response = tool_infer(msg, stream=True)
for part in response:
    print(part.choices[0].delta.content or "", end='')  # type: ignore
```

```shell
To create an effective team of advisors for solving the AI landing application problem, we need to consider individuals with diverse expertise and skill sets. Here's a suggested team comprising three experts:

1. Data Scientist/Machine Learning Expert: This expert should have extensive knowledge in data analysis, machine learning algorithms, and natural language processing (NLP). They will be responsible for designing the AI models that power the landing application, ensuring they are accurate, efficient, and user-friendly. A candidate like Dr. Ziad Kobti, a renowned Data Scientist and Machine Learning Expert, would be an excellent addition to this team.
2. Full Stack Developer: This expert should have experience in both frontend and backend development, as well as knowledge of integrating AI models into web applications. They will work closely with the data scientist to ensure that the landing application is developed according to best practices and user-friendly design principles. A skilled full stack developer like Angela Yu, who has worked on various projects involving AI integration, would be a valuable addition to this team.
3. UX/UI Designer: This expert should have an eye for aesthetics and usability in digital products. They will collaborate with the data scientist and full-stack developer to create an intuitive user interface that enhances the overall experience of users interacting with the AI landing application. A talented designer like Sarah Dunn, who has worked on numerous projects involving UX/UI design for AI applications, would be a great fit for this team.

Together, these three experts will bring their unique skills and perspectives to create an effective and user-friendly AI landing application that meets the desired goals.
```


#### Four、Use Custom Prompt

Above is the default tool call prompt word, Of course you can modify and use it as such：

```python
from tool_llama import Use_Tool
toolkit = Use_Tool.from_template("./your_prompt.toml")
```

the Custom Prompt you can define by a toml file. The following is the default prompt: 

```toml
# your_prompt.toml
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





