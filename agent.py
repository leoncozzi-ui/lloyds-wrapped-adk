import io
import uuid
import altair as alt
from google.adk import Workflow, Event, Context, Agent
from google.adk.tools import google_search
from google.genai import types
from google.cloud import geminidataanalytics
from google.api_core import client_options
from google.protobuf.json_format import MessageToDict

# =====================================================================
# 1. Intent Classifier Agent (LLM Router)
# =====================================================================
classifier_agent = Agent(
    name="classifier_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are an orchestrator that routes user questions. 
    Analyze the user's message and classify it into exactly one of two categories:
    - 'TRANSACTIONS': If the user is asking about Lloyds customer spending, transactions, app sessions, features visited, cashback, or anything specific to the Lloyds banking customer dataset.
    - 'SEARCH': If the user is asking general questions, web search queries, news, weather, calculations, or anything else not related to the Lloyds customer dataset.
    
    You must output exactly the word 'TRANSACTIONS' or 'SEARCH'. Do not output any other text, explanation, or punctuation.
    """
)

def save_input_node(node_input: str):
    """
    Helper node that runs first to save the user's original query 
    in the session state before we pass the query to the classifier agent.
    """
    return Event(state={"original_query": node_input}, output=node_input)

def route_decision(node_input: str, ctx: Context):
    """
    Helper node that reads the classification result and routes to the 
    appropriate branch, forwarding the user's original query as the output.
    """
    intent = node_input.strip().upper()
    original_query = ctx.state.get("original_query", "")
    
    if "TRANSACTIONS" in intent:
        return Event(route="TRANSACTIONS", output=original_query)
    else:
        return Event(route="SEARCH", output=original_query)

# =====================================================================
# 2. General Internet Search Agent (with Google Search Grounding)
# =====================================================================
search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are a professional, helpful assistant with access to real-time information via Google Search.
    Answer the user's question accurately using the search results. 
    Always cite your sources by providing links to the sites you got the information from.
    """,
    tools=[google_search]
)

# =====================================================================
# 3. Lloyds Transaction Analytics Node (BigQuery Client Worker)
# =====================================================================
async def query_lloyds_agent(node_input: str, ctx: Context):
    """
    Custom workflow node that communicates with the BigQuery Conversational Analytics API.
    It manages stateful conversation sessions and handles text, follow-up questions,
    and Vega-Lite chart rendering (converting charts to inline PNGs).
    """
    billing_project = "edb-hack2026-team6"
    location = "us"
    data_agent_id = "agent_8f5e5cf8-79bf-4095-87d1-08477f4a668b"
    
    # Initialize the async chat service client
    endpoint = f"geminidataanalytics.{location}.rep.googleapis.com"
    opts = client_options.ClientOptions(api_endpoint=endpoint)
    client = geminidataanalytics.DataChatServiceAsyncClient(client_options=opts)
    
    # Retrieve or create a persistent BQ conversation name from the session state
    bq_conversation_name = ctx.state.get("bq_conversation_name")
    if not bq_conversation_name:
        conversation_uuid = str(uuid.uuid4())
        conversation_id = f"conv-{conversation_uuid}"
        
        conversation = geminidataanalytics.Conversation()
        conversation.agents = [f"projects/{billing_project}/locations/{location}/dataAgents/{data_agent_id}"]
        conversation.name = f"projects/{billing_project}/locations/{location}/conversations/{conversation_id}"
        
        create_request = geminidataanalytics.CreateConversationRequest(
            parent=f"projects/{billing_project}/locations/{location}",
            conversation_id=conversation_id,
            conversation=conversation,
        )
        
        conversation_resource = await client.create_conversation(request=create_request)
        bq_conversation_name = conversation_resource.name
        
        # Persist the conversation resource name across turns
        yield Event(state={"bq_conversation_name": bq_conversation_name})
        
    # Construct the chat request containing the user's message
    messages = [geminidataanalytics.Message()]
    messages[0].user_message.text = node_input
    
    conversation_reference = geminidataanalytics.ConversationReference()
    conversation_reference.conversation = bq_conversation_name
    conversation_reference.data_agent_context.data_agent = f"projects/{billing_project}/locations/{location}/dataAgents/{data_agent_id}"
    
    chat_request = geminidataanalytics.ChatRequest(
        parent=f"projects/{billing_project}/locations/{location}",
        messages=messages,
        conversation_reference=conversation_reference,
    )
    
    # Stream response from the BQ agent
    stream = await client.chat(request=chat_request)
    
    async for response in stream:
        sys_msg = response.system_message
        if not sys_msg:
            continue
            
        # Convert protobuf to dict for robust attribute access
        sys_msg_dict = MessageToDict(sys_msg._pb)
        
        # 1. Handle Text Messages (Final Responses & Follow-up Questions)
        if "text" in sys_msg_dict:
            text_info = sys_msg_dict["text"]
            text_parts = text_info.get("parts", [])
            text_content = "".join(text_parts)
            text_type = text_info.get("textType", "FINAL_RESPONSE")
            
            if text_type == "FINAL_RESPONSE":
                yield Event(message=text_content)
            elif text_type == "FOLLOWUP_QUESTIONS":
                # Render follow-up suggestions as a neat bulleted list
                followups_md = "\n\n**Suggested Questions:**\n" + "\n".join(f"- {q}" for q in text_parts)
                yield Event(message=followups_md)
                
        # 2. Handle Chart Visualizations (Vega-Lite configurations)
        if "chart" in sys_msg_dict:
            chart_info = sys_msg_dict["chart"]
            vega_config = chart_info.get("result", {}).get("vegaConfig")
            if vega_config:
                try:
                    # Convert the Vega-Lite JSON to a PNG image using Altair
                    chart = alt.Chart.from_dict(vega_config)
                    buf = io.BytesIO()
                    chart.save(buf, format='png')
                    image_bytes = buf.getvalue()
                    
                    # Yield the PNG image as a binary Part in the Event content
                    part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                    yield Event(content=types.Content(parts=[part]))
                except Exception as chart_err:
                    yield Event(message=f"\n*(Error rendering visualization: {chart_err})*\n")

# =====================================================================
# 4. Define the Master Orchestrator Graph
# =====================================================================
root_agent = Workflow(
    name="lloyds_wrapped_orchestrator",
    edges=[
        # Sequential pipeline: Save input -> Classify intent -> Decide the route
        ("START", save_input_node, classifier_agent, route_decision),
        # Conditional branching based on the route decision
        (route_decision, {
            "TRANSACTIONS": query_lloyds_agent,
            "SEARCH": search_agent,
        })
    ]
)
