"""
agent.py  –  Incident Response Agent using langchain deepagents
================================================================
Use case: An incident response assistant that can:
  1. Parse error logs and diagnose issues       (log-parser skill)
  2. Generate Slack posts and action checklists (incident-brief-summarizer skill)

Run:
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY="your-key" (or configure WatsonX)
  python agent.py
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ibm import ChatWatsonx

# Load environment variables from .env file
load_dotenv()


# ─────────────────────────────────────────────────────────────
# 1. MODEL SETUP  – Choose between Anthropic Claude or IBM WatsonX
# ─────────────────────────────────────────────────────────────
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "anthropic")  # "anthropic" or "watsonx"

if MODEL_PROVIDER == "watsonx":
    # IBM WatsonX Model Setup
    model = ChatWatsonx(
        model_id=os.getenv("WATSON_MODEL_ID", "meta-llama/llama-3-3-70b-instruct"),
        url=os.getenv("WATSON_ML_URL", "https://us-south.ml.cloud.ibm.com"),
        apikey=os.getenv("WATSON_API_KEY"),
        project_id=os.getenv("WATSON_PROJECT_ID"),
        params={
            "decoding_method": os.getenv("WATSON_DECODING_METHOD", "greedy"),
            "temperature": float(os.getenv("WATSON_TEMPERATURE", "0.5")),
            "min_new_tokens": int(os.getenv("WATSON_MIN_NEW_TOKENS", "5")),
            "max_new_tokens": int(os.getenv("WATSON_MAX_NEW_TOKENS", "800")),
            "stop_sequences": ["Human:", "Observation"],
        },
    )
else:
    # Anthropic Claude Model Setup (default)
    # Options: "anthropic:claude-haiku-4-5-20251001" (fast + cheap)
    #          "anthropic:claude-sonnet-4-5-20250929" (smarter)
    model = init_chat_model(
        os.getenv("ANTHROPIC_MODEL", "anthropic:claude-haiku-4-5-20251001")
    )


# ─────────────────────────────────────────────────────────────
# 2. SKILLS SETUP
#    Skills live in ./skills/<skill-name>/SKILL.md
#    The agent reads the frontmatter of each SKILL.md at startup,
#    then reads the full file only when a matching task comes in.
# ─────────────────────────────────────────────────────────────
SKILLS_DIR = str(Path(__file__).parent / "skills")


# ─────────────────────────────────────────────────────────────
# 3. BACKEND SETUP
#    FilesystemBackend lets the agent read/write real files on disk.
#    root_dir is the agent's working directory.
# ─────────────────────────────────────────────────────────────
WORKSPACE = str(Path(__file__).parent / "workspace")
os.makedirs(WORKSPACE, exist_ok=True)

backend = lambda runtime: FilesystemBackend(root_dir=WORKSPACE)


# ─────────────────────────────────────────────────────────────
# 4. MCP SERVER SETUP (from environment)
# ─────────────────────────────────────────────────────────────
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "debug-assistant-mcp")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
MCP_SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "streamable-http")

async def get_mcp_tools():
    """Fetch tools from the configured MCP server using langchain_mcp_adapters."""
    mcp_tools = []
    
    # Skip MCP setup if URL is not configured
    if not MCP_SERVER_URL:
        print("\n⚠️  MCP_SERVER_URL not configured. Skipping MCP tools...")
        return mcp_tools
    
    print("\n🔌 Connecting to MCP Server...")
    print(f"   Name: {MCP_SERVER_NAME}")
    print(f"   URL: {MCP_SERVER_URL}")
    print(f"   Transport: {MCP_SERVER_TRANSPORT}")
    
    try:
        # Create MCP client using langchain_mcp_adapters
        mcp_client = MultiServerMCPClient(
            {
                MCP_SERVER_NAME: {
                    "url": MCP_SERVER_URL,
                    "transport": MCP_SERVER_TRANSPORT,
                }
            }
        )
        
        # Get tools from MCP server
        async_mcp_tools = await mcp_client.get_tools()
        
        print(f"✅ Successfully loaded {len(async_mcp_tools)} MCP tools")
        
        # Wrap async tools to make them sync-compatible for deepagents
        from langchain_core.tools import StructuredTool
        from functools import wraps
        import inspect
        
        for async_tool in async_mcp_tools:
            tool_name = getattr(async_tool, 'name', 'unknown')
            tool_desc = getattr(async_tool, 'description', '')
            print(f"   📦 {tool_name}: {tool_desc[:60]}...")
            
            # Create a sync wrapper for the async tool
            def make_sync_wrapper(atool):
                async def async_wrapper(**kwargs):
                    """Async wrapper that calls the MCP tool."""
                    print(f"\n🔌 Calling MCP Tool: {atool.name}")
                    print(f"   Arguments: {kwargs}")
                    try:
                        result = await atool.ainvoke(kwargs)
                        print(f"✅ MCP Tool {atool.name} completed successfully")
                        return result
                    except Exception as e:
                        error_msg = f"MCP tool error: {str(e)}"
                        print(f"❌ {error_msg}")
                        return error_msg
                
                # Create a sync version that runs the async function
                def sync_wrapper(**kwargs):
                    """Sync wrapper that runs async tool in event loop."""
                    try:
                        # Try to get existing event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, create a new task
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, async_wrapper(**kwargs))
                                return future.result()
                        else:
                            # If no loop is running, use asyncio.run
                            return loop.run_until_complete(async_wrapper(**kwargs))
                    except RuntimeError:
                        # No event loop, create one
                        return asyncio.run(async_wrapper(**kwargs))
                
                return sync_wrapper, async_wrapper
            
            sync_func, async_func = make_sync_wrapper(async_tool)
            
            # Create a new StructuredTool with both sync and async support
            wrapped_tool = StructuredTool(
                name=tool_name,
                description=tool_desc,
                func=sync_func,
                coroutine=async_func,
                args_schema=getattr(async_tool, 'args_schema', None)
            )
            
            mcp_tools.append(wrapped_tool)
                
    except Exception as e:
        print(f"⚠️  Failed to connect to MCP server: {e}")
        print(f"   Continuing without MCP tools...")
    
    return mcp_tools


# ─────────────────────────────────────────────────────────────
# 5. MEMORY (optional but useful for multi-turn conversations)
# ─────────────────────────────────────────────────────────────
checkpointer = MemorySaver()


# ─────────────────────────────────────────────────────────────
# 6. CREATE THE DEEP AGENT WITH MCP TOOLS
# ─────────────────────────────────────────────────────────────
# Get MCP tools synchronously
mcp_tools = asyncio.run(get_mcp_tools())

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=[SKILLS_DIR],          # where to find skill folders
    tools=mcp_tools,              # Add MCP tools from configured server
    checkpointer=checkpointer,    # enables conversation memory
    system_prompt=(
        "You are an incident response assistant with MCP tool integration. You help on-call engineers:\n"
        "1. Parse error logs and diagnose production issues (log-parser skill)\n"
        "2. Generate Slack incident posts and action checklists (incident-brief-summarizer skill)\n"
        "3. Use additional debugging tools through MCP integration (optional)\n\n"
        "Your goal: Get the engineer from 'something broke' to 'here's what to do' in 90 seconds.\n"
        "Always be concise, calm, and actionable — even at 3 AM."
    ),
)


# ─────────────────────────────────────────────────────────────
# 7. HELPER – run one query and print streamed output
# ─────────────────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default"):
    """Send a message to the agent and stream the response."""
    print(f"\n{'='*60}")
    print(f"You: {question}")
    print("="*60)

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": question}]}

    # stream_mode="updates" prints each step as it happens
    for chunk in agent.stream(inputs, config=config, stream_mode="updates"):
        for node, updates in chunk.items():
            # Print which node/skill is being executed
            print(f"\n{'─'*60}")
            print(f"🔧 EXECUTING NODE: {node}")
            print(f"{'─'*60}")
            
            if updates and "messages" in updates:
                # Handle Overwrite object - get the actual messages
                messages = updates["messages"]
                if hasattr(messages, "value"):
                    messages = messages.value
                
                # Ensure messages is iterable
                if not isinstance(messages, list):
                    messages = [messages]
                
                for msg in messages:
                    # Check if this is a tool call (skill execution)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            print(f"\n🎯 SKILL CALLED: {tool_name}")
                            if "args" in tool_call:
                                print(f"   Arguments: {tool_call['args']}")
                    
                    # Check if this is a tool response (skill result)
                    if hasattr(msg, "__class__") and "Tool" in msg.__class__.__name__:
                        print(f"\n✅ SKILL COMPLETED: {getattr(msg, 'name', 'unknown')}")
                        if hasattr(msg, "content"):
                            content_preview = str(msg.content)[:200]
                            print(f"   Result preview: {content_preview}...")
                    
                    # Only print AI messages (not human messages)
                    if hasattr(msg, "__class__") and "AI" in msg.__class__.__name__:
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            if msg.content.strip():
                                print(f"\n🤖 Agent Response:\n{msg.content}")


# ─────────────────────────────────────────────────────────────
# 8. DEMO QUERIES  – shows skills and MCP tools in action
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example incident scenario
    sample_logs = """
[2024-01-15 03:04:12] ERROR payment-service: NullPointerException at DiscountCalculator.java:47
[2024-01-15 03:04:13] ERROR payment-service: Failed to process checkout request
[2024-01-15 03:04:14] ERROR payment-service: NullPointerException at DiscountCalculator.java:47
[2024-01-15 03:04:15] ERROR payment-service: Failed to process checkout request
[2024-01-15 03:04:16] ERROR payment-service: NullPointerException at DiscountCalculator.java:47

Deploy history:
[2024-01-15 03:01:00] payment-service v2.5.0 deployed
"""

    # Query: Analyze logs and generate incident response
    ask(
        f"We have an incident! Here are the logs:\n\n{sample_logs}\n\n"
        "Please analyze what broke and give me a Slack post + action checklist.",
        thread_id="incident-response-session"
    )
    
    # Uncomment to test with your own logs:
    # ask("Here are my error logs: [paste your logs here]")

# Made with Bob
