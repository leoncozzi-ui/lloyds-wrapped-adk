import io
import uuid
import altair as alt
from google.adk import Workflow, Event, Context
from google.genai import types
from google.cloud import geminidataanalytics
from google.api_core import client_options
from google.protobuf.json_format import MessageToDict

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

# Define the ADK 2.0 Workflow Graph
root_agent = Workflow(
    name="lloyds_wrapped_workflow",
    edges=[
        ("START", query_lloyds_agent)
    ]
)
