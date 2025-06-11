from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Dict, Any, Optional, Union, Mapping, ClassVar, Set
from openai import OpenAI
from pydantic import Field, PrivateAttr
import os
import json
from datetime import datetime

class LLMClient(BaseChatModel):
    """Custom LLM client using Nebius AI"""
    
    # Define parameters to exclude from API calls
    EXCLUDED_PARAMS: ClassVar[Set[str]] = {
        'callbacks',
        'tags',
        'metadata',
        'run_id',
        'invoke_tags',
        'run_name',
        'execution_order'
    }

    # Private attributes
    _client: OpenAI = PrivateAttr(default=None)
    _retry_count: int = PrivateAttr(default=0)
    _max_retries: int = PrivateAttr(default=2)

    # Required LangChain fields
    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    # Add api_key as a Field
    api_key: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, api_key: str = None, **kwargs):
        """Initialize the LLM client"""
        # First initialize the parent class
        super().__init__(**kwargs)
        # Then set the API key
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        if not self.api_key:
            raise ValueError("Nebius API key is required")
        self._client = self._create_client()
        self._current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def _create_client(self) -> OpenAI:
        """Create OpenAI client for Nebius"""
        return OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=self.api_key
        )

    def _convert_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Convert various message formats to OpenAI format"""
        converted = []
        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage, AIMessage)):
                role = {
                    HumanMessage: "user",
                    SystemMessage: "system",
                    AIMessage: "assistant"
                }.get(type(message), "user")
                converted.append({
                    "role": role,
                    "content": message.content
                })
            elif isinstance(message, dict) and "role" in message and "content" in message:
                converted.append(message)
            else:
                converted.append({
                    "role": "user",
                    "content": str(message)
                })
        return converted

    def _clean_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unsupported parameters from kwargs"""
        return {
            k: v for k, v in kwargs.items()
            if k not in self.EXCLUDED_PARAMS
        }

    async def _agenerate(self, *args, **kwargs) -> ChatResult:
        """Async generate not implemented"""
        raise NotImplementedError("Async generation not supported")

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response and return as ChatResult"""
        try:
            # Convert messages and clean kwargs
            converted_messages = self._convert_messages(messages)
            clean_kwargs = self._clean_kwargs(kwargs)
            if stop:
                clean_kwargs["stop"] = stop

            # Make API call
            response = self._make_api_call(converted_messages, **clean_kwargs)
            
            # Convert response to ChatResult
            if isinstance(response, dict) and "error" in response:
                content = json.dumps(response)
            else:
                content = str(response)

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=content),
                        text=content
                    )
                ]
            )
        except Exception as e:
            print(f"Error in _generate: {e}")
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=str(e)),
                        text=str(e)
                    )
                ]
            )

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Make API call with retry logic"""
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                **kwargs
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            return {"error": "No content in response"}

        except Exception as e:
            print(f"Error with API call: {e}")
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                return self._make_api_call(messages, **kwargs)
            return {
                "error": f"Failed after {self._max_retries} retries",
                "details": str(e),
                "timestamp": self._current_time
            }

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Direct API call method"""
        try:
            converted_messages = self._convert_messages(messages)
            clean_kwargs = self._clean_kwargs({})
            response = self._make_api_call(converted_messages, **clean_kwargs)
            if not response:
                raise ValueError("Empty response from LLM")
            if isinstance(response, dict) and "error" in response:
                raise ValueError(response["error"])
            
            print(f"[LLMClient] Raw LLM response: {repr(response)}")
        
            # If response is already a string, return it
            if isinstance(response, str):
                return response
        
            # If response is a dict, convert it to string
            if isinstance(response, dict):
                if "error" in response:
                    return json.dumps(response)
                return response.get("content", str(response))
            
            # Otherwise, convert to string
            return str(response)
            
        except Exception as e:
            print(f"Error in generate: {e}")
            return json.dumps({
                "error": str(e),
                "metadata": {
                "timestamp": self._current_time,
                "model": self.model_name
                }
            })

    @property
    def _llm_type(self) -> str:
        """Required by LangChain"""
        return "nebius_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters for serialization"""
        return {"model_name": self.model_name}

    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True