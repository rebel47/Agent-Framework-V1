# 📚 Code Explanation Guide

**Complete walkthrough of how the Agent Framework works - explained in simple language**

---

## 📖 Table of Contents

1. [Overview - What Does This Framework Do?](#overview)
2. [Main Components](#main-components)
3. [main.py - The Core Brain](#mainpy-explanation)
4. [api_server.py - The Web Interface](#api_serverpy-explanation)
5. [agents_config.yaml - Agent Configuration](#yaml-configuration)
6. [How Everything Works Together](#how-it-works-together)
7. [Step-by-Step Flow Examples](#step-by-step-examples)

---

## Overview

### What Does This Framework Do?

This framework lets you create **AI agents** that can:
- **Talk to users** (chat)
- **Use tools** (like calculators, web searches, etc.)
- **Remember conversations** (using PostgreSQL database)
- **Work together** (hierarchical multi-agent system)
- **Use different AI models** (Azure OpenAI, Anthropic, Ollama, etc.)

Think of it like this:
- **You** = The user asking questions
- **Agents** = Smart AI assistants that answer you
- **Tools** = Special functions agents can use (like a calculator)
- **Framework** = The system that connects everything

---

## Main Components

### 1. **main.py** (1206 lines)
The **brain** of the framework. Contains all the logic for:
- Creating agents
- Managing conversations
- Loading tools
- Connecting to databases
- Handling different LLM providers

### 2. **api_server.py** (827 lines)
The **web interface** that exposes everything via REST API:
- Chat endpoints
- Agent listing
- OpenAI-compatible endpoints
- Swagger UI documentation

### 3. **agents_config.yaml**
The **configuration file** where you define:
- What agents you want
- What tools each agent can use
- Agent personalities (system prompts)
- Hierarchical relationships

### 4. **tools_registry.py**
Where you define **custom tools** (functions) that agents can use.

---

## main.py Explanation

Let's break down the main.py file section by section:

### 🔧 Section 1: Imports and Setup (Lines 1-120)

```python
# Lines 1-30: Documentation and imports
"""
AgentFramework - Production-Ready AI Multi-Agent System
Built with LangChain v1 + LangGraph + Multi-LLM Support + PostgreSQL
"""
```

**What's happening:**
- Import all necessary libraries (LangChain, FastAPI, database tools, etc.)
- Set up logging to track what the framework is doing
- Configure Windows-specific fixes for async operations

**Why we need this:**
- LangChain = Library for building AI agents
- Logging = To see what's happening (debugging)
- Windows fixes = PostgreSQL async operations need special setup on Windows

---

### 📋 Section 2: Configuration Classes (Lines 120-250)

#### **LLMProvider Enum**
```python
class LLMProvider(str, Enum):
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # etc...
```

**What it does:** Lists all supported AI providers (Azure, OpenAI, Anthropic, etc.)

**Why we need it:** Users can choose which AI service to use

---

#### **LLMProviderConfig Class**
```python
@dataclass
class LLMProviderConfig:
    provider: LLMProvider
    temperature: float = 0.7
    azure_endpoint: Optional[str] = None
    # etc...
```

**What it does:** Stores configuration for connecting to different AI providers

**Why we need it:** Each AI provider needs different settings (API keys, endpoints, etc.)

**Key Methods:**

1. **`from_env()`** - Loads settings from `.env` file
   ```python
   @classmethod
   def from_env(cls) -> 'LLMProviderConfig':
       provider_str = os.getenv("LLM_PROVIDER", "azure")
       # Read all settings from environment variables
   ```
   
   **Example:** If your `.env` has `LLM_PROVIDER=azure`, this loads Azure settings

2. **`validate()`** - Checks if required settings are present
   ```python
   def validate(self) -> None:
       if not value:
           raise ValueError(f"{env_var} not set in .env file.")
   ```
   
   **Example:** If using Azure but no deployment name set → Error!

---

### 🔌 Section 3: MCP Context Provider (Lines 250-350)

```python
class MCPContextProvider:
    """
    Dynamic MCP integration supporting multiple MCP servers.
    Configuration loaded from mcp_servers.yaml
    """
```

**What is MCP?** Model Context Protocol - a way to give agents external information/context

**What it does:**
- Reads `mcp_servers.yaml` configuration
- Connects to external MCP servers
- Fetches additional context for agent responses

**Key Methods:**

1. **`__init__(config_path)`** - Loads MCP configuration
   ```python
   def __init__(self, config_path: str = "mcp_servers.yaml"):
       self.mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
       self._load_config(config_path)
   ```

2. **`get_context(query)`** - Fetches context from MCP servers
   ```python
   async def get_context(self, query: str) -> str:
       # Connect to MCP servers and get relevant context
   ```

**Example use case:** Agent needs weather data → MCP server provides it

---

### 👤 Section 4: Human-in-the-Loop (Lines 350-400)

```python
class HumanInTheLoopMiddleware:
    """
    HITL Middleware for agent supervision.
    Allows human approval/rejection of agent actions.
    """
```

**What it does:** Asks human for approval before agent takes actions

**Why we need it:** Safety! Sometimes you want to approve actions before they happen

**Key Method:**
```python
async def should_proceed(self, action: str, tool_name: str) -> bool:
    response = input("[HITL] Approve? (y/n): ").strip().lower()
    return response == 'y'
```

**Example:**
- Agent wants to send email → HITL asks you first → You approve/reject

---

### 🧠 Section 5: AgentFramework Class (Lines 400-900)

This is the **MAIN CLASS** - the heart of everything!

#### **Initialization (`__init__`)** (Lines 400-450)

```python
def __init__(self, agents_config_path: str = "agents.yaml", tools_loader=None):
    # 1. Load LLM configuration from .env
    self.llm_config = LLMProviderConfig.from_env()
    
    # 2. Get PostgreSQL URL for conversation memory
    self.postgres_url = os.getenv("POSTGRES_URL")
    
    # 3. Initialize MCP and HITL components
    self.mcp = MCPContextProvider()
    self.hitl = HumanInTheLoopMiddleware(...)
    
    # 4. Load agent configurations from YAML
    self.agent_configs = self._load_agent_configs(agents_config_path)
    
    # 5. Storage for created agents
    self.agents: Dict[str, any] = {}
```

**What's happening:**
1. Read settings from `.env` file (which AI to use, database URL, etc.)
2. Create MCP and HITL components
3. Read `agents_config.yaml` to see what agents you want
4. Prepare empty storage for agents (created later in `initialize_async`)

---

#### **Loading Agent Configs (`_load_agent_configs`)** (Lines 450-480)

```python
def _load_agent_configs(self, config_path: str) -> List[Dict]:
    # 1. Open agents_config.yaml file
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 2. Get list of agents
    agents = config.get('agents', [])
    
    # 3. Get default agent name
    self.default_agent_name = config.get('default_agent', None)
    
    # 4. Filter to enabled agents only
    enabled_agents = [a for a in agents if a.get('enabled', True)]
    
    return enabled_agents
```

**What it does:** Reads your YAML file and extracts agent definitions

**Example YAML:**
```yaml
agents:
  - name: "my_assistant"
    description: "Helpful assistant"
    enabled: true  # ← This agent will be loaded
  - name: "old_agent"
    enabled: false  # ← This agent will be skipped
```

---

#### **Initializing LLM (`_initialize_llm`)** (Lines 480-600)

```python
def _initialize_llm(self, temperature: float = None) -> BaseChatModel:
    provider = self.llm_config.provider
    
    if provider == LLMProvider.AZURE:
        return self._init_azure_llm(temperature)
    elif provider == LLMProvider.OPENAI:
        return self._init_openai_llm(temperature)
    # ... etc for other providers
```

**What it does:** Creates the AI language model based on your configuration

**Example flow:**
1. User sets `LLM_PROVIDER=azure` in `.env`
2. Framework calls `_initialize_llm()`
3. Framework detects Azure → calls `_init_azure_llm()`
4. Returns Azure OpenAI model ready to use

**Sub-methods for each provider:**

```python
def _init_azure_llm(self, temperature: float) -> BaseChatModel:
    return AzureChatOpenAI(
        azure_endpoint=self.llm_config.azure_endpoint,
        api_key=self.llm_config.azure_api_key,
        azure_deployment=self.llm_config.azure_deployment,
        temperature=temperature,
        streaming=True,
    )
```

**Why separate methods?** Each AI provider has different parameters and setup

---

#### **Creating Delegation Tool (`_create_delegation_tool`)** (Lines 600-650)

This is for **hierarchical agents** - supervisors delegating to sub-agents.

```python
def _create_delegation_tool(self, supervisor_name: str, sub_agents: List[str]):
    """
    Create a tool that allows a supervisor to delegate tasks to sub-agents.
    """
    
    @tool
    async def delegate_task(task: str, agent_name: str) -> str:
        """Delegate a task to a specialist sub-agent."""
        result = await framework.delegate_to_agent(
            task=task,
            target_agent=agent_name,
            session_id="delegation_session",
            supervisor_agent=supervisor_name
        )
        return result
    
    return delegate_task
```

**What it does:**
1. Creates a special tool for supervisor agents
2. This tool lets them delegate work to sub-agents
3. Example: "General agent" delegates "code review" to "code_analyzer" agent

**Why we need it:** Lets agents work together in teams (hierarchical structure)

---

#### **Async Initialization (`initialize_async`)** (Lines 650-750)

This is where agents actually get created!

```python
async def initialize_async(self):
    # 1. Create PostgreSQL checkpointer (for conversation memory)
    self.checkpointer = await AsyncPostgresSaver.from_conn_string(self.postgres_url)
    await self.checkpointer.setup()
    
    # 2. Loop through each agent config and create agents
    for agent_config in self.agent_configs:
        agent_name = agent_config['name']
        temperature = agent_config.get('temperature', 0.7)
        tool_imports = agent_config.get('tool_imports', [])
        system_prompt = agent_config.get('system_prompt', None)
        sub_agents = agent_config.get('sub_agents', [])
        
        # 3. Load tools for this agent
        agent_tools = _load_tools_from_imports(tool_imports)
        
        # 4. If agent has sub-agents, add delegation tool
        if sub_agents:
            delegation_tool = self._create_delegation_tool(agent_name, sub_agents)
            agent_tools.append(delegation_tool)
        
        # 5. Create LLM instance
        llm = self._initialize_llm(temperature=temperature)
        
        # 6. Create the agent using LangChain
        agent = create_agent(
            model=llm,
            tools=agent_tools,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            debug=debug,
        )
        
        # 7. Store agent in dictionary
        self.agents[agent_name] = {
            'agent': agent,
            'config': agent_config,
            'tools': agent_tools,
            'sub_agents': sub_agents,
        }
```

**Step-by-step breakdown:**

**Step 1:** Connect to PostgreSQL database
- Why? To remember conversation history across sessions

**Step 2-7:** For each agent in your config:
- Load the tools it needs
- Create its AI model (LLM)
- Add delegation tool if it's a supervisor
- Create the agent with LangChain's `create_agent`
- Store it in `self.agents` dictionary

**Example:**
```yaml
# Your config:
agents:
  - name: "helper"
    tools: ["calculator", "weather"]
    sub_agents: ["specialist"]
```

**What happens:**
1. Framework loads "calculator" and "weather" tools
2. Sees "helper" has sub-agent "specialist"
3. Creates delegation tool: "delegate_to_specialist"
4. Creates "helper" agent with 3 tools: calculator, weather, delegate_to_specialist
5. Stores in `self.agents['helper']`

---

#### **Running Conversations (`run_conversation`)** (Lines 800-920)

This is where the magic happens - actually talking to agents!

```python
async def run_conversation(
    self, 
    user_message: str, 
    session_id: str = "default",
    agent_name: Optional[str] = None
):
    # 1. Get the agent to use
    agent_data = self.get_agent(agent_name)
    agent = agent_data['agent']
    
    # 2. Get MCP context if enabled
    mcp_context = await self.mcp.get_context(user_message)
    
    # 3. Check HITL approval if enabled
    if agent_data.get('hitl', False):
        if not await self.hitl.should_proceed(...):
            return "Request rejected by human supervisor."
    
    # 4. Prepare config with session ID (for conversation memory)
    config = {
        "configurable": {
            "thread_id": f"{session_id}",
        },
        "recursion_limit": 100
    }
    
    # 5. Run the agent
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        version="v2",
    ):
        pass  # Events are processed internally
    
    # 6. Get final response
    final_state = await agent.aget_state(config)
    response_content = final_state.values["messages"][-1].content
    
    return response_content
```

**Flow explained:**

**Step 1:** Find the agent
- If you don't specify, uses default agent
- Example: `agent_name="code_analyzer"` → uses that agent

**Step 2:** Get external context
- If MCP enabled → fetch additional information
- Example: User asks "weather in NYC" → MCP provides weather data

**Step 3:** HITL check
- If Human-in-the-Loop enabled → ask for approval
- Example: Agent wants to delete file → you approve first

**Step 4:** Setup conversation config
- `thread_id` = session identifier (like "user_123")
- Same thread_id = agent remembers previous messages
- Different thread_id = new conversation, no memory

**Step 5:** Run the agent
- Sends user message to AI
- Agent thinks and decides if it needs tools
- Agent might call tools multiple times
- Agent generates final response

**Step 6:** Extract and return response
- Get the last message from conversation
- Return it to user

**Example conversation:**
```
User: "What's 15 * 23?"
↓
Framework: Calls run_conversation()
↓
Agent: Sees user wants calculation
Agent: Uses calculator tool: calculate(15 * 23)
Agent: Gets result: 345
Agent: Generates response: "15 × 23 = 345"
↓
Framework: Returns "15 × 23 = 345" to user
```

---

#### **Delegation to Sub-Agents (`delegate_to_agent`)** (Lines 750-800)

For hierarchical agent systems - when one agent delegates to another.

```python
async def delegate_to_agent(
    self,
    task: str,
    target_agent: str,
    session_id: str,
    supervisor_agent: Optional[str] = None
) -> str:
    # 1. Validate target agent exists
    if target_agent not in self.agents:
        raise ValueError(f"Target agent '{target_agent}' not found")
    
    # 2. Validate delegation authority
    if supervisor_agent:
        sub_agents = self.get_sub_agents(supervisor_agent)
        if target_agent not in sub_agents:
            raise ValueError(f"Agent '{supervisor_agent}' cannot delegate to '{target_agent}'")
    
    # 3. Create sub-session ID
    sub_session_id = f"{session_id}_delegated_{target_agent}"
    
    # 4. Run the sub-agent
    response = await self.run_conversation(
        user_message=task,
        session_id=sub_session_id,
        agent_name=target_agent
    )
    
    return response
```

**What it does:** Safely delegates work from one agent to another

**Example scenario:**
```
User: "Review this Python code and optimize it"
↓
Main Agent (supervisor): "This needs code expertise, I'll delegate"
↓
Framework: delegate_to_agent(task="Review code...", target_agent="code_analyzer")
↓
Code Analyzer Agent: Analyzes code, finds issues
↓
Framework: Returns code review to Main Agent
↓
Main Agent: "Here's what the code analyzer found: ..."
```

**Safety checks:**
1. **Agent exists?** Can't delegate to non-existent agent
2. **Authority check:** Only supervisors can delegate to their sub-agents
3. **Separate session:** Sub-agent gets its own session ID to avoid confusion

---

### 🎯 Section 6: Helper Functions (Lines 920-1206)

#### **create_cli()** - Command Line Interface

```python
def create_cli(agents_config_path: str = "agents.yaml"):
    class CLIChat:
        async def initialize(self):
            # Create framework
            self.framework = AgentFramework(agents_config_path=...)
            await self.framework.initialize_async()
        
        async def chat_loop(self):
            while True:
                user_input = input("👤 You: ")
                
                # Smart routing - pick best agent for query
                selected_agent = self.router.route(user_input, available_agents)
                
                # Get response
                response = await self.framework.run_conversation(
                    user_message=user_input,
                    agent_name=selected_agent
                )
                
                print(f"🤖 [{selected_agent}]: {response}")
```

**What it does:** Creates an interactive chat in terminal

**Features:**
- **Smart routing:** Automatically picks best agent for your question
- **Commands:** Type "help", "agents", "quit"
- **Conversation memory:** Remembers context within session

**Example usage:**
```bash
$ python agents.py
👤 You: What's 5 + 5?
🤖 [calculator_agent]: 5 + 5 = 10

👤 You: What's the weather?
🤖 [weather_agent]: It's sunny, 75°F
```

---

#### **create_api()** - Web API Interface

```python
def create_api(agents_config_path: str = "agents.yaml", host: str = "0.0.0.0", port: int = 8000):
    class APIServer:
        def run(self):
            from api_server import run_api_server
            run_api_server(host=self.host, port=self.port, config_path=self.config_path)
```

**What it does:** Starts a web server with REST API

**Usage:**
```python
# In your agents.py:
api = create_api()
api.run()  # Starts server on http://localhost:8000
```

---

## api_server.py Explanation

The API server provides web access to your agents.

### 🌐 Section 1: Data Models (Lines 1-160)

#### **ChatRequest**
```python
class ChatRequest(BaseModel):
    message: str  # What user wants to say
    thread_id: Optional[str]  # Conversation ID (optional)
    agent_name: Optional[str]  # Which agent to use (optional)
```

**What it does:** Defines the structure of chat requests

**Example JSON:**
```json
{
    "message": "Hello, agent!",
    "thread_id": "user_123_session",
    "agent_name": "my_assistant"
}
```

---

#### **ChatResponse**
```python
class ChatResponse(BaseModel):
    response: str  # Agent's reply
    thread_id: str  # Conversation ID
    agent_name: str  # Which agent answered
```

**What it does:** Defines the structure of responses

**Example JSON:**
```json
{
    "response": "Hello! How can I help you?",
    "thread_id": "user_123_session",
    "agent_name": "my_assistant"
}
```

---

#### **AgentInfo**
```python
class AgentInfo(BaseModel):
    name: str  # Agent's name
    system_prompt: Optional[str]  # Agent's personality
    tools: List[str]  # Available tools
    temperature: float  # Creativity setting
    sub_agents: Optional[List[str]]  # Sub-agents if supervisor
    parent: Optional[str]  # Parent agent if sub-agent
    role: str  # "supervisor", "sub-agent", or "agent"
```

**What it does:** Provides detailed agent information

**Example JSON:**
```json
{
    "name": "main_agent",
    "system_prompt": "You are a helpful assistant",
    "tools": ["calculator", "search"],
    "temperature": 0.7,
    "sub_agents": ["specialist_agent"],
    "parent": null,
    "role": "supervisor"
}
```

---

### 🚀 Section 2: Application Lifecycle (Lines 160-220)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global framework
    framework = AgentFramework(agents_config_path=config_path)
    await framework.initialize_async()
    
    yield  # Run the application
    
    # SHUTDOWN
    if framework:
        await framework.cleanup()
```

**What it does:**
- **On startup:** Create and initialize framework
- **While running:** Framework available to all endpoints
- **On shutdown:** Clean up database connections

**Why we need it:** Framework must be ready before handling requests

---

### 📡 Section 3: API Endpoints (Lines 220-650)

#### **GET /health** - Check if server is running

```python
@app.get("/health")
async def health_check():
    return HealthResponse(
        status="healthy",
        framework_initialized=True,
        agents_count=len(framework.list_agents()),
        llm_provider=framework.llm_config.provider.value,
    )
```

**What it does:** Returns server status

**Example response:**
```json
{
    "status": "healthy",
    "framework_initialized": true,
    "agents_count": 3,
    "llm_provider": "azure"
}
```

**Use case:** Check if API is ready before making requests

---

#### **GET /agents** - List all agents

```python
@app.get("/agents")
async def list_agents():
    agents = framework.list_agents()
    return AgentListResponse(
        agents=agents,
        default_agent=framework.default_agent_name,
        total_count=len(agents),
    )
```

**What it does:** Returns list of available agents

**Example response:**
```json
{
    "agents": ["helper", "code_analyzer", "researcher"],
    "default_agent": "helper",
    "total_count": 3
}
```

**Use case:** Discover what agents are available

---

#### **GET /agents/{agent_name}** - Get agent details

```python
@app.get("/agents/{agent_name}")
async def get_agent_info(agent_name: str):
    # Find agent config
    agent_config = # ... find in framework.agent_configs
    
    # Determine role
    if agent_config.get('sub_agents'):
        role = "supervisor"
    elif agent_config.get('parent'):
        role = "sub-agent"
    else:
        role = "agent"
    
    return AgentInfo(
        name=agent_config['name'],
        tools=agent_config.get('tools', []),
        role=role,
        # ... etc
    )
```

**What it does:** Returns detailed info about specific agent

**Example request:** `GET /agents/code_analyzer`

**Example response:**
```json
{
    "name": "code_analyzer",
    "system_prompt": "You are an expert code reviewer...",
    "tools": ["analyze_code", "lint_code"],
    "temperature": 0.3,
    "role": "sub-agent",
    "parent": "main_assistant"
}
```

**Use case:** See what a specific agent can do

---

#### **GET /agents/hierarchy/tree** - View agent hierarchy

```python
@app.get("/agents/hierarchy/tree")
async def get_agent_hierarchy():
    supervisors = []
    standalone_agents = []
    
    # Build hierarchy tree
    for config in framework.agent_configs:
        if config.get('sub_agents'):
            supervisors.append({
                "name": config['name'],
                "sub_agents": config['sub_agents'],
                "role": "supervisor"
            })
        elif not config.get('parent'):
            standalone_agents.append({
                "name": config['name'],
                "role": "agent"
            })
    
    return {
        "supervisors": supervisors,
        "standalone_agents": standalone_agents,
    }
```

**What it does:** Shows supervisor → sub-agent relationships

**Example response:**
```json
{
    "supervisors": [
        {
            "name": "main_agent",
            "sub_agents": ["code_analyzer", "researcher"],
            "role": "supervisor"
        }
    ],
    "standalone_agents": [
        {
            "name": "helper",
            "role": "agent"
        }
    ],
    "supervisor_count": 1,
    "standalone_count": 1
}
```

**Use case:** Understand how agents are organized

---

#### **POST /chat** - Send message to agent

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Generate thread_id if not provided
    thread_id = request.thread_id or f"api_{id(request)}"
    
    # 2. Use specified agent or default
    agent_name = request.agent_name or framework.default_agent_name
    
    # 3. Verify agent exists
    if agent_name not in framework.list_agents():
        raise HTTPException(404, detail=f"Agent '{agent_name}' not found")
    
    # 4. Run conversation
    response_text = await framework.run_conversation(
        user_message=request.message,
        session_id=thread_id,
        agent_name=agent_name,
    )
    
    # 5. Return response
    return ChatResponse(
        response=response_text,
        thread_id=thread_id,
        agent_name=agent_name,
    )
```

**What it does:** Main endpoint for chatting with agents

**Example request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is 5 * 7?",
    "thread_id": "session_123",
    "agent_name": "calculator_agent"
  }'
```

**Example response:**
```json
{
    "response": "5 multiplied by 7 equals 35.",
    "thread_id": "session_123",
    "agent_name": "calculator_agent"
}
```

**Flow:**
1. Receive user message
2. Find the right agent
3. Call `framework.run_conversation()`
4. Return agent's response

---

#### **POST /chat/stream** - Streaming responses

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        # Get response from agent
        response_text = await framework.run_conversation(...)
        
        # Stream word by word
        words = response_text.split()
        for word in words:
            yield f"data: {word}\n\n"
            await asyncio.sleep(0.05)
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**What it does:** Sends response word-by-word in real-time

**Use case:** Better user experience - see response as it's generated

**Format:** Server-Sent Events (SSE)

---

### 🔌 Section 4: OpenAI-Compatible Endpoints (Lines 550-750)

These endpoints make the framework compatible with Open WebUI, TypingMind, LibreChat, etc.

#### **GET /v1/models** - List models (agents)

```python
@app.get("/v1/models")
async def openai_list_models():
    agents = framework.list_agents()
    models = [
        OpenAIModelInfo(
            id=agent_name,  # Agent name becomes model ID
            created=int(time.time())
        )
        for agent_name in agents
    ]
    return {"object": "list", "data": models}
```

**What it does:** Lists agents as if they were OpenAI models

**Example response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "helper",
            "object": "model",
            "created": 1699123456,
            "owned_by": "agentframework"
        },
        {
            "id": "code_analyzer",
            "object": "model",
            "created": 1699123456,
            "owned_by": "agentframework"
        }
    ]
}
```

**Use case:** Open WebUI shows these as selectable models

---

#### **POST /v1/chat/completions** - OpenAI chat format

```python
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    # 1. Extract user message from OpenAI format
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    user_message = user_messages[-1].content
    
    # 2. Map "model" to agent name
    agent_name = request.model or framework.default_agent_name
    
    # 3. Generate thread_id from message history
    thread_id = f"openai_{hash(''.join([m.content for m in request.messages]))}"
    
    # 4. Run conversation
    response_text = await framework.run_conversation(
        user_message=user_message,
        session_id=thread_id,
        agent_name=agent_name
    )
    
    # 5. Return in OpenAI format
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": agent_name,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }
```

**What it does:** Accepts OpenAI-format requests, returns OpenAI-format responses

**Example OpenAI request:**
```json
{
    "model": "code_analyzer",
    "messages": [
        {"role": "user", "content": "Review this code: def add(a, b): return a+b"}
    ]
}
```

**Example OpenAI response:**
```json
{
    "id": "chatcmpl-a1b2c3d4",
    "object": "chat.completion",
    "model": "code_analyzer",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The code looks good! The function correctly adds two numbers."
            },
            "finish_reason": "stop"
        }
    ]
}
```

**Why we need this:** Makes framework work with existing OpenAI-compatible tools

---

## YAML Configuration

### agents_config.yaml Structure

```yaml
# Optional: Specify default agent
default_agent: "general_assistant"

agents:
  # Agent 1: Basic agent
  - name: "general_assistant"
    description: "General purpose AI assistant"
    system_prompt: |
      You are a helpful AI assistant.
      Be concise but thorough.
    tools:
      - "get_current_time"
      - "calculate"
    enabled: true
    temperature: 0.7
    debug: false
    
  # Agent 2: Supervisor with sub-agents
  - name: "main_supervisor"
    description: "Main coordinator agent"
    system_prompt: "You coordinate tasks and delegate to specialists."
    tools:
      - "get_current_time"
    sub_agents:
      - "code_analyzer"
      - "researcher"
    enabled: true
    temperature: 0.5
    
  # Agent 3: Sub-agent
  - name: "code_analyzer"
    description: "Code review specialist"
    system_prompt: "You are an expert code reviewer."
    tools:
      - "lint_code"
      - "analyze_complexity"
    parent: "main_supervisor"
    enabled: true
    temperature: 0.3
    debug: true
```

### Field Explanations

#### **name** (required)
- **Type:** String
- **What it is:** Unique identifier for the agent
- **Example:** `"code_analyzer"`
- **Used for:** Selecting agent in API calls

#### **description** (required)
- **Type:** String
- **What it is:** Brief description of agent's purpose
- **Example:** `"Analyzes and reviews code"`
- **Used for:** Smart routing (framework picks agent based on description)

#### **system_prompt** (optional)
- **Type:** String (can be multi-line with `|`)
- **What it is:** Agent's personality/instructions
- **Example:**
  ```yaml
  system_prompt: |
    You are a friendly teacher.
    Explain concepts in simple terms.
    Use examples and analogies.
  ```
- **Used for:** Controls how agent responds

#### **tools** (optional)
- **Type:** List of strings
- **What it is:** Names of tools this agent can use
- **Example:**
  ```yaml
  tools:
    - "calculator"
    - "web_search"
    - "send_email"
  ```
- **Used for:** Gives agent capabilities beyond just chatting

#### **tool_imports** (optional, newer method)
- **Type:** List of import strings
- **What it is:** Explicit Python imports for tools
- **Example:**
  ```yaml
  tool_imports:
    - "tools_registry.get_current_time"
    - "my_tools.custom_analyzer"
  ```
- **Why better:** More explicit, avoids naming conflicts

#### **sub_agents** (optional)
- **Type:** List of strings
- **What it is:** Names of agents this agent can delegate to
- **Example:**
  ```yaml
  name: "main_agent"
  sub_agents:
    - "code_specialist"
    - "data_specialist"
  ```
- **Used for:** Hierarchical agent structures
- **Effect:** Automatically creates delegation tool for this agent

#### **parent** (optional)
- **Type:** String
- **What it is:** Name of parent (supervisor) agent
- **Example:**
  ```yaml
  name: "specialist"
  parent: "main_agent"
  ```
- **Used for:** Documentation and hierarchy visualization
- **Note:** Must match a sub_agent entry in parent's config

#### **enabled** (optional, default: true)
- **Type:** Boolean
- **What it is:** Whether to load this agent
- **Example:** `enabled: false`
- **Used for:** Temporarily disable agents without deleting config

#### **temperature** (optional, default: 0.7)
- **Type:** Float (0.0 to 1.0)
- **What it is:** Controls randomness/creativity
- **Examples:**
  - `0.0` = Very deterministic, same answer every time
  - `0.3` = Low creativity (good for code, math)
  - `0.7` = Balanced (default)
  - `1.0` = Very creative (good for stories, ideas)

#### **debug** (optional, default: false)
- **Type:** Boolean
- **What it is:** Shows agent's thinking process
- **Example:** `debug: true`
- **Output when true:**
  ```
  [Agent thinking] I need to use calculator tool...
  [Tool call] calculator(5 * 7)
  [Tool result] 35
  [Agent response] The answer is 35
  ```

#### **hitl** (optional, default: false)
- **Type:** Boolean
- **What it is:** Human-in-the-loop - ask for approval
- **Example:** `hitl: true`
- **Effect:** Agent asks for permission before actions

---

## How Everything Works Together

### Complete Request Flow

Let's trace a complete request from user to response:

#### Example: User asks "What's 15 * 23?" via API

**Step 1: API receives request**
```
POST /chat
{
    "message": "What's 15 * 23?",
    "agent_name": "general_assistant"
}
```

**Step 2: api_server.py processes it**
```python
@app.post("/chat")
async def chat(request: ChatRequest):
    # Extract data
    message = "What's 15 * 23?"
    agent_name = "general_assistant"
    thread_id = "api_12345"
    
    # Call framework
    response = await framework.run_conversation(
        user_message=message,
        session_id=thread_id,
        agent_name=agent_name
    )
```

**Step 3: main.py framework processes it**
```python
async def run_conversation(self, user_message, session_id, agent_name):
    # 1. Get agent
    agent_data = self.get_agent("general_assistant")
    agent = agent_data['agent']
    
    # 2. Prepare config with session ID
    config = {
        "configurable": {"thread_id": "api_12345"},
        "recursion_limit": 100
    }
    
    # 3. Run agent
    await agent.astream_events(
        {"messages": [HumanMessage(content="What's 15 * 23?")]},
        config=config
    )
```

**Step 4: LangChain agent processes it**
```
Agent receives: "What's 15 * 23?"
Agent thinks: "This is a math problem, I need my calculator tool"
Agent calls: calculate("15 * 23")
```

**Step 5: Tool executes**
```python
@tool
def calculate(expression: str) -> str:
    result = eval(expression)
    return str(result)

# Returns: "345"
```

**Step 6: Agent generates response**
```
Agent receives tool result: "345"
Agent thinks: "I'll format this nicely for the user"
Agent generates: "15 multiplied by 23 equals 345."
```

**Step 7: Framework returns response**
```python
# main.py
response_content = "15 multiplied by 23 equals 345."
return response_content
```

**Step 8: API server formats response**
```python
# api_server.py
return ChatResponse(
    response="15 multiplied by 23 equals 345.",
    thread_id="api_12345",
    agent_name="general_assistant"
)
```

**Step 9: User receives response**
```json
{
    "response": "15 multiplied by 23 equals 345.",
    "thread_id": "api_12345",
    "agent_name": "general_assistant"
}
```

---

### Hierarchical Agent Flow

Example: User asks main agent to review code, which delegates to code_analyzer

**User request:**
```
"Review this Python code: def add(a, b): return a+b"
```

**Flow:**

1. **Main agent receives request**
   ```
   Agent: "main_supervisor"
   Thinks: "This is a code review task, I should delegate to code_analyzer"
   ```

2. **Main agent uses delegation tool**
   ```python
   # Agent calls its delegation tool
   delegate_task(
       task="Review this Python code: def add(a, b): return a+b",
       agent_name="code_analyzer"
   )
   ```

3. **Framework validates delegation**
   ```python
   # In delegate_to_agent():
   # Check: Is "code_analyzer" in main_supervisor's sub_agents? ✓ Yes
   # Check: Does "code_analyzer" agent exist? ✓ Yes
   # Proceed with delegation
   ```

4. **Code analyzer agent processes**
   ```
   Agent: "code_analyzer"
   Receives: "Review this Python code: def add(a, b): return a+b"
   Uses: lint_code tool, analyze_complexity tool
   Response: "Code looks good! Simple addition function with clear parameters."
   ```

5. **Response flows back up**
   ```
   code_analyzer → main_supervisor → User
   ```

6. **User sees:**
   ```
   "I delegated to my code specialist. Here's their review:
   Code looks good! Simple addition function with clear parameters."
   ```

---

### Conversation Memory Flow

How the framework remembers conversations:

**Session 1:**
```
User: "My name is John"
Agent: "Nice to meet you, John!"

# Stored in PostgreSQL:
thread_id: "user_123"
messages:
  - User: "My name is John"
  - Assistant: "Nice to meet you, John!"
```

**Session 2 (same thread_id):**
```
User: "What's my name?"
# Agent loads history from PostgreSQL
# Sees: "My name is John" from before
Agent: "Your name is John!"
```

**Session 3 (different thread_id):**
```
User: "What's my name?"
# Agent has no history for this thread
Agent: "I don't know your name yet."
```

**How it works:**
```python
# In run_conversation:
config = {
    "configurable": {
        "thread_id": "user_123"  # ← This is the key
    }
}

# LangChain automatically:
# 1. Loads previous messages with this thread_id from PostgreSQL
# 2. Adds new message to conversation
# 3. Saves updated conversation back to PostgreSQL
```

---

## Step-by-Step Examples

### Example 1: Adding a New Tool

**Goal:** Add a weather tool that agents can use

**Step 1:** Create tool in `tools_registry.py`
```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city
    
    Returns:
        Weather description
    """
    # In production, call real weather API
    return f"Weather in {city}: Sunny, 75°F"
```

**Step 2:** Add to agent config in `agents_config.yaml`
```yaml
agents:
  - name: "general_assistant"
    tools:
      - "get_weather"  # ← Add this line
      - "calculate"
```

**OR** use explicit imports:
```yaml
agents:
  - name: "general_assistant"
    tool_imports:
      - "tools_registry.get_weather"  # ← Better approach
      - "tools_registry.calculate"
```

**Step 3:** Restart framework
```bash
python agents.py
```

**Step 4:** Test it
```
User: "What's the weather in NYC?"
Agent: Uses get_weather tool
Agent: "Weather in NYC: Sunny, 75°F"
```

---

### Example 2: Creating a Hierarchical Agent System

**Goal:** Create a main agent with two specialists

**Step 1:** Define agents in `agents_config.yaml`
```yaml
agents:
  # Main supervisor
  - name: "assistant"
    description: "General assistant that coordinates tasks"
    system_prompt: "You coordinate tasks and delegate to specialists when needed."
    tools:
      - "get_current_time"
    sub_agents:  # ← This makes it a supervisor
      - "coder"
      - "writer"
    enabled: true
    
  # Specialist 1
  - name: "coder"
    description: "Expert in programming and code analysis"
    system_prompt: "You are an expert programmer. Analyze and write code."
    tools:
      - "run_code"
      - "lint_code"
    parent: "assistant"  # ← Links to supervisor
    enabled: true
    
  # Specialist 2
  - name: "writer"
    description: "Expert in writing and editing text"
    system_prompt: "You are an expert writer. Create and improve text."
    tools:
      - "check_grammar"
      - "improve_text"
    parent: "assistant"  # ← Links to supervisor
    enabled: true
```

**Step 2:** Framework automatically creates delegation tool
```python
# When framework sees sub_agents, it creates:
@tool
async def delegate_to_sub_agent(task: str, agent_name: str) -> str:
    """Delegate task to coder or writer"""
    # ... delegation logic
```

**Step 3:** Use it
```
User: "Write a Python function to calculate factorial"

assistant thinks: "This is a coding task, delegate to coder"
↓
assistant calls: delegate_to_sub_agent(task="...", agent_name="coder")
↓
coder executes: Uses run_code tool, creates function
↓
coder returns: "Here's the factorial function: def factorial(n): ..."
↓
assistant returns: "I had my coding specialist create this for you: ..."
```

---

### Example 3: Configuring Different LLM Providers

**Scenario:** Switch from Azure OpenAI to local Ollama

**Current .env (Azure):**
```bash
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

**New .env (Ollama):**
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**That's it!** Framework automatically:
1. Reads `LLM_PROVIDER=ollama`
2. Calls `_init_ollama_llm()` instead of `_init_azure_llm()`
3. Connects to local Ollama server
4. Uses llama3.2 model

**No code changes needed!**

---

## Summary

### Key Concepts

1. **AgentFramework** = The main class that manages everything
2. **Agents** = AI assistants with specific capabilities
3. **Tools** = Functions agents can call (calculator, search, etc.)
4. **YAML Config** = Where you define agents without coding
5. **API Server** = Web interface to interact with agents
6. **PostgreSQL** = Database for conversation memory
7. **Hierarchical System** = Supervisors can delegate to sub-agents

### File Responsibilities

| File | Responsibility |
|------|---------------|
| `main.py` | Core framework logic, agent creation, conversation handling |
| `api_server.py` | REST API endpoints, web interface |
| `agents_config.yaml` | Agent definitions and configuration |
| `tools_registry.py` | Custom tool definitions |
| `.env` | API keys, database URL, provider selection |

### Common Tasks

**Want to...** | **Do this...**
---|---
Add new agent | Add entry to `agents_config.yaml`
Add new tool | Define in `tools_registry.py`, add to agent's tools list
Change AI provider | Update `LLM_PROVIDER` in `.env`
Enable debug mode | Set `debug: true` in agent config
Create agent hierarchy | Use `sub_agents` and `parent` in YAML
Enable human approval | Set `hitl: true` in agent config
Change agent personality | Update `system_prompt` in YAML
Use API instead of CLI | Run with `--api` flag

### Request Flow Summary

```
User Input
    ↓
API Server (api_server.py)
    ↓
AgentFramework (main.py)
    ↓
Specific Agent (LangChain)
    ↓
[Uses Tools if needed]
    ↓
LLM (Azure/OpenAI/etc.)
    ↓
Agent Response
    ↓
API Server
    ↓
User Output
```

---

## 🎯 Next Steps

Now that you understand the code:

1. **Experiment:** Try modifying `agents_config.yaml`
2. **Create Tools:** Add custom tools in `tools_registry.py`
3. **Test Hierarchies:** Create supervisor-subordinate agent structures
4. **Try Different LLMs:** Switch between Azure, Ollama, Anthropic
5. **Build Something:** Use the framework for your own project!

**Questions?** Check the other documentation files:
- `DEVELOPER_GUIDE.md` - Advanced topics
- `DEBUG_GUIDE.md` - Troubleshooting
- `HIERARCHICAL_AGENTS.md` - More on multi-agent systems
