from pydantic import BaseModel
from typing import Dict, Any, Type


def get_model_schema(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.model_json_schema()
    function = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": pydantic_class.model_json_schema(),
    }
    return function
