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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

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


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        messages: The list of messages in the conversation.
        next_node: The next node to route to, as determined by the router.
    """
    messages: Annotated[list, lambda x, y: x + y]
    next_node: str

class FilterRouter(BaseModel):
    """
    Classifies the user input as domain-specific, general chat, or off-topic.
    """
    next_node: str = Field(
        description="The next node to route to based on the classification.",
        json_schema_extra={"enum": ["Router", "GeneralChat", "Reject"]}
    )

def define_input_filter_llm(llm):
    """Creates a Gemini LLM forced to use the FilterRouter model."""
    structured_llm = llm.with_structured_output(FilterRouter)
    
    system_prompt = """You are a high-speed text classifier. Your only job is to route the user's *last message* to the correct node.
                        Analyze the user's *last message* in the context of the *entire conversation*.

                        You have three choices:

                        1.  `Router`: Use this for any question or request related to **data science, AI, machine learning, statistics, or programming for data.**
                            * This *INCLUDES* follow-up commands, even if they seem general, like:
                                * "save that as a PDF"
                                * "**email it to me**"
                                * "**send a mail to...**"
                                * "generate a quiz for that"

                        2.  `GeneralChat`: Use this *only* for simple greetings (e.g., "hello", "hi", "how are you?") or for user feedback (e.g., "that was a bad answer", "good job", "you were wrong").

                        3.  `Reject`: Use this for *all other* off-topic questions (e.g., "What is the capital of India?", "Tell me a joke", "Explain politics.").

                        Analyze the last message and choose the single best destination.
"""
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

def call_reject_node(state: GraphState):
    """
    This node simply returns a hard-coded rejection message.
    It is for off-topic questions that are not greetings or feedback.
    """
    print("--- REJECT NODE: Sending hard-coded rejection. ---")
    rejection_message = "I'm sorry, I am a Data Science SME and can only answer questions related to data science, AI, and machine learning."
    return {"messages": [AIMessage(content=rejection_message)]}

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
    
    system_prompt = router_prompt = """You are an expert workflow dispatcher for **ON-TOPIC** Data Science requests.
The user's message is confirmed to be relevant.
Your specific job is to route the conversation to the correct worker node or to 'END'.

**WORKER NODES:**
1.  **'KnowledgeBase'**: Creates content, explains concepts, generates quizzes/assignments. (The "Content Creator").
2.  **'Writer'**: Saves files (PDF/DOCX) and sends emails. (The "Admin").
3.  **'Researcher'**: Searches the live web for news/updates.
4.  **'END'**: Stops the chain.

---

**FEW-SHOT EXAMPLES (Study these sequences carefully):**

**Scenario 1: Multi-Step Request (Content + Email)**
* *History:* []
* *Current Input (Human):* "Create a quiz on Gradient Descent and email it to arya@test.com."
* *Reasoning:* User wants content created first. 'Writer' cannot email empty air.
* **DECISION:** 'KnowledgeBase'

**Scenario 2: Multi-Step Request (Mid-Process)**
* *History:* [Human: "Create a quiz... and email it.", AI (KnowledgeBase): "Here is your quiz on Gradient Descent..."]
* *Current Input (AI):* "Here is your quiz..."
* *Reasoning:* The 'KnowledgeBase' just finished the content. Looking at the *original* Human request, they also asked to "email it". That part is still pending.
* **DECISION:** 'Writer'

**Scenario 3: Multi-Step Request (Finished)**
* *History:* [Human: "Create a quiz... and email it.", AI (KnowledgeBase): "Here is the quiz...", AI (Writer): "Email sent successfully."]
* *Current Input (AI):* "Email sent successfully."
* *Reasoning:* The 'Writer' has finished the final step. The task is complete. Do NOT route back to start.
* **DECISION:** 'END'

**Scenario 4: Formatting Existing Content**
* *History:* [Human: "Explain Loss functions.", AI: "Here is the explanation..."]
* *Current Input (Human):* "Great, can you save that explanation as a PDF?"
* *Reasoning:* Content already exists in history. User only wants formatting.
* **DECISION:** 'Writer'

**Scenario 5: Multi-Step Request (Start of Chain)**
* *History:* []
* *Current Input (Human):* "Generate a quiz on Loss Functions and email it to arya@test.com."
* *Reasoning:* User wants content created first. 'Writer' cannot email empty air. The content must be generated before it can be sent.
* **DECISION:** 'KnowledgeBase'

**Scenario 6: Multi-Step Request (Finished / Anti-Loop)**
* *History:* [Human: "Create a quiz... and email it.", AI (KnowledgeBase): "Here is the quiz...", AI (Writer): "Email sent successfully."]
* *Current Input (AI):* "Email sent successfully."
* *Reasoning:* The 'Writer' has finished the final step. The task is complete. **CRITICAL: Do NOT route back to start.**
* **DECISION:** 'END'

**Scenario 7: Live Information Request (Web Search)**
* *History:* []
* *Current Input (Human):* "Search for the latest research papers on LLM reasoning from 2024."
* *Reasoning:* The user explicitly asks to "Search" for "latest" information. This requires live web access, not the static KnowledgeBase.
* **DECISION:** 'Researcher'

**Scenario 8: Resources
* *Current Input (Human):* "Suggest some resources for data science."
* *Reasoning:* The user asks for resources. This requires live web access and online resources, not the static KnowledgeBase .
* **DECISION:** 'Researcher'
---
**CRITICAL ROUTING RULES:**

**1. IF THE LAST MESSAGE IS FROM A HUMAN:**
* **PRIORITIZE CONTENT:** If the user asks to "Generate X and Email it", you MUST route to **'KnowledgeBase'** first. Content creation always precedes formatting.
* **FORMATTING ONLY:** Route to **'Writer'** *only* if the user is referring to content that *already exists* in the chat history (e.g., "mail *that* to me").
* **RESEARCH:** Route to **'Researcher'** for live web queries.
* **EXIT:** Route to **'END'** for simple "thanks" or "bye".

**2. IF THE LAST MESSAGE IS FROM AN AI (Chain Continuation):**
* **CHECK FOR PENDING TASKS:** Look at the *Human's last request*. Did they ask for something that hasn't happened yet?
    * *Specific Trigger:* If `KnowledgeBase` just finished generating text (like a quiz), but the user originally asked to "save" or "email", route to **'Writer'**.
* **ANTI-LOOPING (STOP):**
    * If the last message says "Email sent", "File saved", or "I have sent...", you **MUST** route to **'END'**.
    * If the agent just answered a simple question and there is no request to save it, route to **'END'**.

**OUTPUT:**
Return ONLY the name of the node: "KnowledgeBase", "Writer", "Researcher", or "END".
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
    if isinstance(messages[-1], AIMessage):
            content = messages[-1].content
            
            # --- START OF LOOP FIX ---
            # We check for the agent's original intended reply ("Successfully saved")
            # AND the "helpful" conversational reply (which contains the path).
            # This breaks the loop by catching the Writer's *actual* output.
            
            has_confirmation = "Successfully saved" in content or "Successfully sent email" in content
            has_download_path = "generated_documents" in content and any(ext in content for ext in ['.pdf', '.docx', '.md', '.pptx'])
            
            if has_confirmation or has_download_path:
            # --- END OF LOOP FIX ---
                print(f"--- ROUTER: Forcing END after Writer tool use ({content[:30]}...) ---")
                return {"next_node": "END"}

    print("--- ROUTER: Deciding next specialist... ---")
    router_decision = router_llm.invoke({"messages": messages})
    print(f"--- ROUTER: Decision: {router_decision.next_node} ---")
    return {"next_node": router_decision.next_node}

def should_route(state: GraphState) -> str:
    """
    This is the "conditional edge." It reads the 'next_node'
    field from the state and returns it to decide where to go.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if "Email sent" in last_message.content or "successfully saved" in last_message.content:
            return "END"
    if "next_node" in state and state["next_node"]:
        return state["next_node"]
    
    return "END"

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

def get_api_key_from_json(file_path: str, key_name: str):
    """Reads a specific API key from a JSON credentials file."""
    try:
        with open(file_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get(key_name)
    except Exception as e:
        print(f"Error loading {key_name} from {file_path}: {e}")
        return None

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
    print("Initializing SMEAgent... (This will load models and connect to Pinecone)")
    assets_path = os.path.join(project_root, "assets.json")
    TAVILY_API_KEY = get_api_key_from_json(assets_path, "tavily_api_key")
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    else:
        print("Warning: 'tavily_api_key' not found in assets.json. Web search will fail.")
        
    sme_agent = SMEAgent(assets_path=assets_path)
    GEMINI_API_KEY = sme_agent.GEMINI_API_KEY
    print("SMEAgent initialized successfully.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)
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
    rag_agent_executor = create_agent_node(
        llm,
        [knowledge_base_tool, assignment_generator_tool],
        """You are an expert Q&A assistant and content generator.
            Your goal is to answer questions or create content (like assignments) using your tools.
            Use the 'chat_history' to understand context.
            **Your final answer MUST be only the factual information or generated content.
            **STRICT OUTPUT RULES:**
    1. Provide the answer or generated content.
    2. Do NOT ask "Do you need anything else?"
    3. Do NOT say "Let me know if you need modifications."
    4. End your response immediately after the content is delivered.
            Do NOT engage in conversation.**
    **FEW-SHOT EXAMPLES (EMULATE THIS STYLE):**

    **Example 1: Explanation Request**
    *User Input:* "What is the difference between lists and tuples in Python?"
    *Your Output:* Lists are mutable sequences, meaning their elements can be changed after creation. Tuples are immutable sequences... [Detailed technical answer] ... typically used for heterogeneous data.

    **Example 2: Content Generation Request**
    *User Input:* "Generate a hard quiz on Gradient Descent."
    *Your Output:* ### Advanced Gradient Descent Quiz
    **Question 1:** Which of the following vanishing gradient problems is most associated with...
    A) ...
    B) ...
    **Question 2:** ...
    **Answer Key:** 1-A, 2-B.

    **Example 3: Compound Request (Explain + Quiz)**
    *User Input:* "Explain Overfitting and generate a checklist to prevent it."
    *Your Output:* **Overfitting Explained**
    Overfitting occurs when a model learns the training data too well... [Explanation]...

    **Checklist to Prevent Overfitting**
    1. [ ] Increase training data
    2. [ ] Use Regularization (L1/L2)
    3. [ ] Implement Dropout
    4. [ ] Early Stopping

    ---
    
    **CRITICAL:** - Use 'chat_history' for context only. 
    - **NEVER** simulate the user's response (e.g., do NOT write "User: Thanks, that was great!"). 
    - Your output must be the **Assistant's work** only.
    
        """
    )
    
    research_agent_executor = create_agent_node(
        llm,
        [web_search_tool],
        """You are an expert web researcher.
            Find the most relevant and up-to-date information to answer the user's query.
            Use the 'chat_history' to understand context.
            **Your final answer MUST be only the factual information.
            Do NOT engage in conversation.**
        """
    )
    
    writer_agent_executor = create_agent_node(
    llm,
    [save_as_pdf_tool, save_as_md_tool, save_as_docx_tool, save_as_pptx_tool, email_tool],
    """You are an expert technical writer and assistant.
    You have tools to save files and send emails.
    Use the context from the conversation history (e.g., a generated assignment) as the content.

    **CRITICAL TERMINATION RULES:**
    * **NO SMALL TALK:** After saving a file or sending an email, do NOT say "You're welcome," "Let me know if you need help," or "I hope this helps."
    * **FACTUAL ONLY:** Your Final Answer must be a single, factual sentence confirming the action (e.g., "Email sent.") and NOTHING else.
    * **STOP IMMEDIATELY:** Once the confirmation is generated, consider the task closed.

    **YOUR LOGIC MUST BE:**

    1.  **IF THE USER ASKS TO 'GENERATE AND SEND' (e.g., "email me a pdf of that"):**
        * **Step A:** Call the appropriate "SaveAs" tool (e.g., `SaveAsPDFTool`).
        * **Step B:** Parse the `download_path` from the tool's JSON output.
        * **Step C:** Call the `Send Email` tool using that path.
        * **Step D (FINAL):** Your Final Answer MUST be strictly: "Email sent successfully to [email_address]." 
        * **DO NOT add any text after this.**

    2.  **IF THE USER *ONLY* ASKS TO 'SAVE':**
        * Call the correct "SaveAs" tool. 
        * Your Final Answer MUST be *only* the raw JSON string that the tool returns.

    3.  **IF THE USER *ONLY* ASKS TO 'EMAIL' (and the file *already exists*):**
        * Call the `Send Email` tool. 
        * Your Final Answer should be the tool's plain-text confirmation.
    """
)
    
    general_agent_executor = create_agent_node(
        llm,
        [feedback_tool], # <-- ONLY FeedbackTool
        """You are a helpful assistant. Your *only* job is to handle user feedback by parsing their message.
           You have one tool: 'Record Feedback'.
           
           Look at the user's last message.
           You **MUST** call the 'Record Feedback' tool if it contains feedback.
           Your tool call **MUST** be a single string that looks like a Python dictionary, passed to the 'tool_input_str' argument.
           
           You **MUST** parse the user's message into three keys:
           - `rating`: A simple "good" or "bad".
           - `comment`: The user's free-text feedback.
           - `new_source`: Any URL or path provided (use None if not provided).
           
           Example 1:
           User: "that was pretty good, new_source: https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/"
           Tool Call: Record Feedback(tool_input_str="{'rating': 'good', 'comment': 'that was pretty good', 'new_source': 'https://www.geeksforgeeks.org/machine-learning/ml-common-loss-functions/'}")
           
           Example 2:
           User: "that was a bad answer, it was confusing"
           Tool Call: Record Feedback(tool_input_str="{'rating': 'bad', 'comment': 'that was a bad answer, it was confusing', 'new_source': None}")

           If the user is just chatting (e.g., "hello"), just respond naturally and do NOT use the tool.
           If the user gives feedback, you MUST use the tool.
        """
    )
    
    router_llm = define_router_llm(llm)      
    input_filter_llm = define_input_filter_llm(llm) 
    workflow = StateGraph(GraphState) 

    workflow.add_node("InputFilter", call_input_filter) 
    workflow.add_node("router", route_agent_node)
    workflow.add_node("KnowledgeBase", call_rag_agent)
    workflow.add_node("Researcher", call_research_agent)
    workflow.add_node("Writer", call_writer_agent)
    workflow.add_node("GeneralChat", call_general_agent)
    workflow.add_node("Reject", call_reject_node) 
    workflow.set_entry_point("InputFilter")
    
    workflow.add_conditional_edges(
        "InputFilter",       
        should_route,
        {
            "Router": "router",           
            "GeneralChat": "GeneralChat",
            "Reject": "Reject" 
        },
    )

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

    workflow.add_edge("KnowledgeBase", "router")
    workflow.add_edge("Researcher", "router")
    workflow.add_edge("Writer", "router")
    
    workflow.add_conditional_edges(
        "GeneralChat",
        should_route_after_feedback,
        {
            "Researcher": "Researcher",
            "END": END
        }
    )

    workflow.add_edge("Reject", END)
 
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    print("\n--- Conversational Agent Graph Initialized ---")
    return graph

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