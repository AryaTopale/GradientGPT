"""
Conversational Agent Graph for a Subject Matter Expert (SME)
This file defines a multi-agent system using LangGraph.

Architecture:
1.  State: A 'GraphState' object holds the conversation memory (`messages`)
    and the next node to route to (`next_node`).
2.  InputFilter: A "triage" node (using Gemini) classifies input
    as "on-topic" (Router), "general" (GeneralChat), or "off-topic" (Reject).
3.  Router: A central 'router' node (using Gemini) routes
    *on-topic* requests to the correct specialist.
4.  Worker Agents:
    - KnowledgeBase: Uses RAG (via sme_agent) and Gemini.
    - Researcher: Uses Tavily and Gemini.
    - Writer: Uses file/email tools and Gemini.
    - GeneralChat: A guardrail agent (using Gemini) for
      greetings and feedback.
    - Reject: A "dumb" node to reject off-topic questions.
"""

import os
import json
import ast
from typing import TypedDict, Annotated, List, Union, Dict, Tuple, Optional, Type
import sys
from pydantic import BaseModel, Field
from itertools import cycle  # <-- Import for key rotation

# --- ADDED: Fix for relative import ---
# Add the parent directory (project root) to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- END ADDITION ---

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# --- Local Imports ---
from rag_api.sme_agent import (
    SMEAgent, 
    SaveAsPDFTool, 
    SaveAsMDTool, 
    FeedbackTool,
    MDToolInput,
    PDFToolInput,
    DocxToolInput,
    PptxToolInput,
    SaveAsDocxTool,
    SaveAsPptxTool,
    EmailToolInput,
    EmailTool
)
from langchain_tavily import TavilySearch


# --- 1. Define the State (The System's Memory) ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        messages: The list of messages in the conversation.
        next_node: The next node to route to, as determined by the router.
    """
    messages: Annotated[list, lambda x, y: x + y]
    next_node: str

# --- 2. Define the Input Filter ---
class FilterRouter(BaseModel):
    """
    Classifies the user input as domain-specific, general chat, or off-topic.
    """
    next_node: str = Field(
        description="The next node to route to based on the classification.",
        # --- THIS IS THE 3-WAY TRIAGE ---
        json_schema_extra={"enum": ["Router", "GeneralChat", "Reject"]}
    )

def define_input_filter_llm(llm):
    """Creates a Gemini LLM forced to use the FilterRouter model."""
    structured_llm = llm.with_structured_output(FilterRouter)
    
    # --- UPDATED 3-WAY PROMPT ---
    system_prompt = """You are a high-speed text classifier. Your only job is to route the user's *last message* to the correct node.
Analyze the user's *last message* in the context of the *entire conversation*.

You have three choices:

1.  `Router`: Use this for any question or request related to **data science, AI, machine learning, statistics, or programming for data.**
    * This ALSO includes **follow-up commands** about the previous message, such as "save that as a PDF", "email it to me", or "generate a quiz for that".

2.  `GeneralChat`: Use this *only* for simple greetings (e.g., "hello", "hi", "how are you?") or for user feedback (e.g., "that was a bad answer", "good job", "you were wrong").

3.  `Reject`: Use this for *all other* off-topic questions (e.g., "What is the capital of India?", "Tell me a joke", "Explain politics.").

Analyze the last message and choose the single best destination.
"""
    # --- END OF UPDATE ---

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])
    return prompt | structured_llm

def call_input_filter(state: GraphState):
    """
    This is the new entry point. It classifies the user input
    and decides where to route.
    """
    messages = state['messages']
    print("--- INPUT FILTER: Classifying user input... ---")
    decision = input_filter_llm.invoke({"messages": messages})
    print(f"--- INPUT FILTER: Decision: {decision.next_node} ---")
    
    # Store the decision in 'next_node' for the conditional edge
    return {"next_node": decision.next_node}

# --- 3. Define the Worker Agents ---

def create_agent_node(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    """Helper function to create a new agent node."""
    
    prompt = hub.pull("hwchase17/react-chat")
    prompt = prompt.partial(
        instructions=system_prompt,
        tools=tools,
        tool_names=", ".join([t.name for t in tools]),
    )
    
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

# --- 4. Define the Graph Nodes ---
def call_rag_agent(state: GraphState):
    """Node for the RAG (Knowledge Base) agent."""
    messages = state['messages']
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = rag_agent_executor.invoke(input_data)
    return {"messages": [AIMessage(content=response['output'])]}

def call_research_agent(state: GraphState):
    """Node for the Web Research agent."""
    messages = state['messages']
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = research_agent_executor.invoke(input_data)
    return {"messages": [AIMessage(content=response['output'])]}

def call_writer_agent(state: GraphState):
    """Node for the Writing & Saving agent."""
    messages = state['messages']
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = writer_agent_executor.invoke(input_data)
    return {"messages": [AIMessage(content=response['output'])]}

def call_general_agent(state: GraphState):
    """Node for the General Chatbot (now with FeedbackTool)."""
    messages = state['messages']
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = general_agent_executor.invoke(input_data)
    return {"messages": [AIMessage(content=response['output'])]}

# --- *** ADD THIS NEW "DUMB" NODE *** ---
def call_reject_node(state: GraphState):
    """
    This node simply returns a hard-coded rejection message.
    It is for off-topic questions that are not greetings or feedback.
    """
    print("--- REJECT NODE: Sending hard-coded rejection. ---")
    rejection_message = "I'm sorry, I am a Data Science SME and can only answer questions related to data science, AI, and machine learning."
    return {"messages": [AIMessage(content=rejection_message)]}
# --- *** END OF NEW FUNCTION *** ---


# --- 5. Define the Main Router (for ON-TOPIC requests) ---
class Router(BaseModel):
    """
    Route the ON-TOPIC conversation to the next appropriate specialist.
    """
    next_node: str = Field(
        description="The name of the next specialist node to route to.",
        json_schema_extra={"enum": ["KnowledgeBase", "Researcher", "Writer", "END"]}
    )

def define_router_llm(llm: ChatGoogleGenerativeAI):
    """Creates an LLM with tool-calling forced to use the Router model."""
    structured_llm = llm.with_structured_output(Router)
    
    system_prompt = """You are an expert dispatcher for **ON-TOPIC** data science requests.
                        The user's message has *already* been classified as on-topic.
                        Your job is to route it to the correct specialist or to 'END'.

                        **CRITICAL ROUTING RULES:**

                        1.  **IF THE LAST MESSAGE IS AN `AIMessage` (from an agent):**
                            * The task is finished. Your *only* valid choice is "END".
                            * **Exception:** If the `AIMessage` is from 'KnowledgeBase' (content was just generated) AND the user's *original* request *also* asked to 'save' or 'email' it (e.g., "generate an assignment and send it to me"), then you MUST route to 'Writer'.

                        2.  **IF THE LAST MESSAGE IS A `HumanMessage` (which you know is on-topic):**
                            * You must route to the *first* agent that matches the user's intent, in this exact priority order:

                            * **Priority 1: 'Writer'**
                                * Route here if the user's request contains keywords like "save", "pdf", "markdown", "docx", "pptx", or "email".
                                * The model is smart enough to use the content from the chat history.
                            * **Priority 2: 'KnowledgeBase'**
                                * Route here for *new* questions or content generation requests (e.g., "explain loss functions", "make a quiz on regression").
                            * **Priority 3: 'Researcher'**
                                * Route here for *explicit* web research requests (e.g., "what's the latest news on...", "find me information about...").
                            * **Priority 4: 'END'**
                                * Route here if the message is a simple goodbye (e.g., "thanks", "bye").

                        Choose the *single best* node based on these priorities.
                        """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    router_chain = prompt | structured_llm
    return router_chain

def route_agent_node(state: GraphState):
    """
    This node runs the *main* router (after filtering)
    and decides which specialist to call.
    """
    messages = state['messages']
    last_message = messages[-1]

    # --- *** START OF LOOP FIX *** ---
    if isinstance(last_message, AIMessage):
        # We assume KB answers are long, and Writer/GeneralChat answers are short.
        is_kb_answer = len(last_message.content) > 150 
        
        if not is_kb_answer:
            # This message is a short confirmation. Force the turn to end.
            print(f"--- ROUTER: Short AIMessage detected. Forcing END. ---")
            return {"next_node": "END"}
        
        print(f"--- ROUTER: Long KB answer detected. Letting LLM router decide... ---")
    # --- *** END OF LOOP FIX *** ---

    # If the last message is a HumanMessage, let the LLM router decide.
    print("--- ROUTER: HumanMessage detected, deciding next specialist... ---")
    router_decision = router_llm.invoke({"messages": messages})
    print(f"--- ROUTER: Decision: {router_decision.next_node} ---") # <-- Fixed typo
    
    return {"next_node": router_decision.next_node}

# --- 6. Define Conditional Edges ---
def should_route(state: GraphState) -> str:
    """
    This is the "conditional edge." It reads the 'next_node'
    field from the state and returns it to decide where to go.
    """
    return state["next_node"]

def should_route_after_feedback(state: GraphState) -> str:
    """
    This conditional edge runs AFTER the GeneralChat node.
    It checks the user's *original* message (now at index -2)
    to see if it contained a URL to be researched.
    """
    print("--- FEEDBACK ROUTER: Checking for follow-up search... ---")
    last_user_message = state['messages'][-2]
    content = last_user_message.content
    
    if "http://" in content or "https://" in content:
        print("--- FEEDBACK ROUTER: Found URL, routing to Researcher. ---")
        return "Researcher"
    else:
        print("--- FEEDBACK ROUTER: No URL found, ending turn. ---")
        return "END"

# --- Helper function to load API keys ---
def get_api_key_from_json(file_path: str, key_name: str):
    """Reads a specific API key from a JSON credentials file."""
    try:
        with open(file_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get(key_name)
    except Exception as e:
        print(f"Error loading {key_name} from {file_path}: {e}")
        return None

# --- 7. Graph Factory Function ---
# Global definitions for the agent executors
rag_agent_executor: AgentExecutor
research_agent_executor: AgentExecutor
writer_agent_executor: AgentExecutor
general_agent_executor: AgentExecutor
router_llm: any
input_filter_llm: any

def get_agent_graph():
    """
    Initializes all components and returns the compiled agent graph.
    """
    global rag_agent_executor, research_agent_executor, writer_agent_executor, general_agent_executor, router_llm, input_filter_llm
    
    print("Initializing...")
    assets_path = os.path.join(project_root, "assets.json")
    
    TAVILY_API_KEY = get_api_key_from_json(assets_path, "tavily_api_key")
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    else:
        print("Warning: 'tavily_api_key' not found in assets.json. Web search will fail.")
        
    # --- *** START OF KEY ROTATION UPDATE *** ---
    
    # 1. Load the LIST of keys from assets.json
    #    (Make sure your assets.json has "GEMINI_API_KEYS": ["key1", "key2", ...])
    GEMINI_API_KEYS = get_api_key_from_json(assets_path, "gemini_api_key")
    if not GEMINI_API_KEYS or not isinstance(GEMINI_API_KEYS, list) or len(GEMINI_API_KEYS) == 0:
        raise ValueError("GEMINI_API_KEYS not found in assets.json, is not a list, or is empty.")

    # 2. Create the LLM Pool
    print(f"Creating LLM pool with {len(GEMINI_API_KEYS)} keys...")
    llm_pool = [
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0)
        for key in GEMINI_API_KEYS
    ]
    # Creates an infinite iterator that cycles through your LLM objects
    llm_cycler = cycle(llm_pool)
    
    # 3. Inject the first LLM from the pool into SMEAgent
    #    (This requires SMEAgent.py to be updated to accept 'llm' in __init__)
    try:
        sme_agent = SMEAgent(llm=next(llm_cycler), assets_path=assets_path)
        print("SMEAgent initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize SMEAgent. Did you update sme_agent.py to accept 'llm' in __init__?")
        print(f"Full error: {e}")
        raise e
    
    # --- *** END OF KEY ROTATION UPDATE *** ---

    
    # --- B. Define REAL Tools based on SMEAgent ---
    @tool
    def knowledge_base_tool(query: str) -> str:
        """
        Use this tool to answer specific questions about data science concepts,
        course material, or internal knowledge.
        """
        return sme_agent.run_qa_rag(query)
    
    @tool
    def assignment_generator_tool(query: str) -> str:
        """
        Use this tool to generate the text content for an assignment, quiz,
        or study guide based on the internal knowledge.
        """
        return sme_agent.run_assignment_rag(query)

    save_as_pdf_tool = Tool(
        name="Save As PDF",
        func=lambda tool_input_str: SaveAsPDFTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to save text content into a .pdf file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
    )
    save_as_md_tool = Tool(
        name="Save As Markdown",
        func=lambda tool_input_str: SaveAsMDTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to save text content into a .md file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
    )
    save_as_docx_tool = Tool(
        name="Save As DOCX",
        func=lambda tool_input_str: SaveAsDocxTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to save text content into a .docx file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
    )
    save_as_pptx_tool = Tool(
        name="Save As PPTX",
        func=lambda tool_input_str: SaveAsPptxTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to save text content into a .pptx presentation file. The input MUST be a Python dictionary-like string with two keys: 'content' and 'filename'."
    )
    email_tool = Tool(
        name="Send Email",
        func=lambda tool_input_str: EmailTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to send an email with a subject, body, and an optional file attachment. Input MUST be a Python dictionary-like string with 'to_email', 'subject', 'body', and optional 'attachment_path'."
    )
    feedback_tool = Tool(
        name="Record Feedback",
        func=lambda tool_input_str: FeedbackTool()._run(
            **ast.literal_eval(tool_input_str)
        ),
        description="Use this tool to record user feedback. The input MUST be a Python dictionary-like string with three keys: 'rating' (e.g., 'good' or 'bad'), 'comment' (the text), and 'new_source' (a URL or None)."
    )
    web_search_tool = TavilySearch(max_results=5)
    web_search_tool.name = "web_search_tool"


    # --- C. Instantiate Worker Agents ---
    # --- UPDATED: Assign a different LLM from the pool to each agent ---
    rag_agent_executor = create_agent_node(
        next(llm_cycler),  # Use next LLM in pool
        [knowledge_base_tool, assignment_generator_tool],
        """You are an expert Q&A assistant and content generator.
            Your *only* job is to answer questions or generate content (like assignments) using your tools.
            Use the 'chat_history' to understand context.
            **CRITICAL RULES:**
            1.  Your final answer MUST be **only** the factual information or the generated content.
            2.  Do NOT engage in conversation. Do NOT say "hello", "sure", or "here is the information".
            3.  You **MUST NOT** talk about file operations. Do NOT mention PDFs, saving, downloading, or emailing. Your job is *only* to produce the text.
        """
    )
    
    research_agent_executor = create_agent_node(
        next(llm_cycler),  # Use next LLM in pool
        [web_search_tool],
        """You are an expert web researcher.
            Find the most relevant and up-to-date information to answer the user's query.
            Use the 'chat_history' to understand context.
            **Your final answer MUST be only the factual information.
            Do NOT engage in conversation.**
        """
    )
    
    writer_agent_executor = create_agent_node(
        next(llm_cycler),  # Use next LLM in pool
        [save_as_pdf_tool, save_as_md_tool, save_as_docx_tool, save_as_pptx_tool, email_tool],
        """You are a file-processing and email assistant.
            You MUST use the context from the chat history to perform your tasks.
            **CRITICAL PATHS:**
            - Your "save" tools (like Save As PDF) will return a *web path* like: `{"download_path": "/generated_documents/filename.pdf", ...}`
            - Your "EmailTool" needs a *filesystem path*.
            - The correct filesystem path is `generated_documents/filename.pdf` (You **MUST remove the leading slash**).
            **TOOL CHAINING EXAMPLE:**
            1.  User: "save and email"
            2.  Thought: First, I must save the content to get a path.
                Action: `Save As PDF` (using content from history)
                Action Input: `{'content': '...text...', 'filename': 'file.pdf'}`
            3.  Observation: `{"download_path": "/generated_documents/file.pdf", ...}`
            4.  Thought: Now I must email. I will strip the leading slash from the path.
                Action: `Send Email`
                Action Input: `{'to_email': '...', 'subject': '...', 'body': '...', 'attachment_path': 'generated_documents/file.pdf'}`
            **FINAL ANSWER RULE:**
            - Your final answer MUST be **ONLY** the JSON string from the "save" tool (if saving).
            - Your final answer MUST be **ONLY** the confirmation string from the "email" tool (if emailing).
            - **DO NOT** say "I have...", "Successfully...", "You're welcome...", or anything else.
        """
    )
    
    general_agent_executor = create_agent_node(
        next(llm_cycler),  # Use next LLM in pool
        [feedback_tool],
        """You are a helpful assistant with two specific jobs, listed by priority.
You will *only* receive messages that are either greetings or feedback.

**PRIORITY 1: Handle User Feedback**
If the message IS feedback (e.g., "that was bad", "good answer", "this is helpful"), your *only* job is to call the `Record Feedback` tool.
Your tool call **MUST** be a single string that looks like a Python dictionary...
(e.g., Tool Call: Record Feedback(tool_input_str="{'rating': 'bad', 'comment': '...'}"))

**PRIORITY 2: Handle Simple Greetings**
If the message is *only* a simple greeting (e.g., "hello", "hi", "how are you"), just respond naturally and do NOT use the tool.
Your `Final Answer:` should be a simple greeting (e.g., "Hello! How can I help you today?").
"""
    )
    
    # --- D. Instantiate Routers ---
    router_llm = define_router_llm(next(llm_cycler))
    input_filter_llm = define_input_filter_llm(next(llm_cycler))

    # --- E. Build the Graph (NEW WIRING) ---
    workflow = StateGraph(GraphState) 

    # 1. Add all nodes
    workflow.add_node("InputFilter", call_input_filter) # New filter node
    workflow.add_node("router", route_agent_node)       # This is the MAIN router
    workflow.add_node("KnowledgeBase", call_rag_agent)
    workflow.add_node("Researcher", call_research_agent)
    workflow.add_node("Writer", call_writer_agent)
    workflow.add_node("GeneralChat", call_general_agent)
    workflow.add_node("Reject", call_reject_node)       # <-- ADDED NEW REJECT NODE
    
    # 2. Set the NEW entry point
    workflow.set_entry_point("InputFilter")
    
    # 3. Add conditional edge from the new filter
    workflow.add_conditional_edges(
        "InputFilter",       
        should_route,
        {
            "Router": "router",           # On-topic -> Go to main router
            "GeneralChat": "GeneralChat", # Greetings/Feedback -> Go to GeneralChat
            "Reject": "Reject"            # <-- ADDED REJECT ROUTE
        },
    )

    # 4. Add conditional edge from the MAIN router
    workflow.add_conditional_edges(
        "router",       
        should_route,
        {
            "KnowledgeBase": "KnowledgeBase",
            "Researcher": "Researcher",
            "Writer": "Writer",
            "END": END,
        },
    )

    # 5. Add edges for worker nodes to loop back
    workflow.add_edge("KnowledgeBase", "router")
    workflow.add_edge("Researcher", "router")
    workflow.add_edge("Writer", "router")
    
    # 6. Add the feedback edge
    workflow.add_conditional_edges(
        "GeneralChat",
        should_route_after_feedback,
        {
            "Researcher": "Researcher",
            "END": END
        }
    )
    
    # 7. --- ADD THIS NEW EDGE ---
    # The Reject node simply ends the graph turn.
    workflow.add_edge("Reject", END)
    # --- *** END OF NEW WIRING *** ---

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer) # <-- Fixed typo
    
    print("\n--- Conversational Agent Graph Initialized ---")
    return graph

# --- 8. Main execution block ---
if __name__ == "__main__":
    
    graph = get_agent_graph()
    
    print("Type 'exit' to quit.")
    conversation_config = {"configurable": {"thread_id": "sme-convo-1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=conversation_config
        )
        
        print("\nAgent:")
        final_message = None
        for event in events:
            if "messages" in event:
                final_message = event['messages'][-1]
        
        if final_message:
            final_message.pretty_print()