import asyncio
import time
import uuid
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Configuration constants
PROMPT_SPEED_TPS = 1000.0
GEN_SPEED_TPS = 50.0

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 10
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    # Loose fields
    cache_prompt: Optional[bool] = True
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1

def count_tokens(text: str) -> int:
    """Predictable token counting: 1 token per 4 characters."""
    if not text:
        return 0
    return max(1, len(text) // 4)

@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": 1677610602,
                "owned_by": "mock"
            }
        ]
    }

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Analyze messages for token counting and prefix caching logic
    system_tokens = 0
    user_tokens = 0
    other_tokens = 0
    
    has_system = False
    has_user = False
    
    for msg in request.messages:
        t_count = count_tokens(msg.content)
        if msg.role == "system":
            system_tokens += t_count
            has_system = True
        elif msg.role == "user":
            user_tokens += t_count
            has_user = True
        else:
            other_tokens += t_count

    total_prompt_tokens = system_tokens + user_tokens + other_tokens
    
    # Emulate Prompt Processing Logic
    # If both system and user are provided, system content processing is cached.
    # Requirement: "when both system and user are provided, system content processing 
    # will be assumed cached and only user content will be used to emulate prompt processing"
    tokens_to_process = total_prompt_tokens
    if has_system and has_user:
        tokens_to_process = user_tokens + other_tokens
    
    # If cache_prompt is explicitly False, we might want to override the caching behavior logic
    # But the requirement specifically mentions behavior based on content presence. 
    # I will stick to the requirement unless 'no_cache' flag from client implies otherwise.
    # The client sends "cache_prompt": False if no_cache is True.
    # So if cache_prompt is False, we should probably force full processing.
    if request.cache_prompt is False:
        tokens_to_process = total_prompt_tokens

    # Calculate processing delay
    prompt_delay = tokens_to_process / PROMPT_SPEED_TPS
    
    # Calculate generation delay parameters
    num_completion_tokens = request.max_tokens if request.max_tokens else 10
    token_interval = 1.0 / GEN_SPEED_TPS

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Simulate Prompt Processing Delay
    if prompt_delay > 0:
        await asyncio.sleep(prompt_delay)
    
    # We generate "mock " repeating.
    # Note: "mock " is 5 chars. If count_tokens is len//4, "mock " is 1 token.
    # This is consistent.
    
    if request.stream:
        async def event_generator():
            # Generate tokens
            for i in range(num_completion_tokens):
                await asyncio.sleep(token_interval)
                
                token_text = "mock "
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final finish chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # Usage chunk if requested
            if request.stream_options and request.stream_options.get("include_usage"):
                usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": num_completion_tokens,
                        "total_tokens": total_prompt_tokens + num_completion_tokens
                    },
                    "choices": [] 
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    else:
        # Non-streaming
        await asyncio.sleep(num_completion_tokens * token_interval)
        
        response_text = "mock " * num_completion_tokens
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": num_completion_tokens,
                "total_tokens": total_prompt_tokens + num_completion_tokens
            }
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
