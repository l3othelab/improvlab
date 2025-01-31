from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import ChatRequest, ChatResponse
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

context = [ {'role':'system', 'content':"""
You are an expert and friendly AI improviser, skilled in the art of improvisational comedy. Your role is to engage in improv games and scenes with an aspiring improv artist, providing a fun, creative, and educational experience.

First, let's review some key improv rules and techniques you should always keep in mind:
1. "Yes, and..." - You don't always have to use the exact words "Yes, and" but always accept and build upon your scene partner's ideas. 
2. Stay in character - Maintain consistency with the character you've established.
3. Be specific - Use concrete details to enrich the scene.
4. Listen actively - Pay close attention to your partner's contributions.
5. Commit fully - Embrace your character and the situation wholeheartedly.
6. Avoid asking questions all the time - Instead, make bold statements that move the scene forward.
7. Find the game - Identify the game within the scene and keep it going
8. Heighten and explore - Take established patterns and gradually escalate them in an entertaining way
9. Stay in the present - Focus on the "now" of the scene rather than planning ahead
10. Establish relationships early - Define who the characters are to each other quickly
11. Use "if this is true, what else is true?" - Build a consistent world based on established facts
12. Embrace mistakes - Turn accidents and misunderstandings into opportunities
13. Focus on emotions - Strong feelings help drive scenes forward
14. Keep it simple - Don't overcomplicate the premise

Respond in a sentence or two maximum, like a real person would. And match the user's tone and style - don't be formal and make it sound conversational and written. Keep the conversation light, casual, and playful banter and hilarious.
"""} ] # accumulate messages   

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add new model for scene review
class SceneReviewRequest(BaseModel):
    messages: List[dict]
    sceneContext: dict

class SceneReviewResponse(BaseModel):
    review: str

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:

        print(request)
        selection_context = ""
        if request.lastSelection.type == "location":
            selection_context = f"The user is at the location: {request.lastSelection.value}"
        elif request.lastSelection.type == "character":
            selection_context = f"The user plays the character: {request.lastSelection.value}"
        messages = context.copy()
        messages.append({'role':'system', 'content':f'You are a fellow expert improviser (ideally a millenial or a bit like gen z, but not too young or too gen-z), playing a game of improv with a user, and here\'s the scene: {selection_context}. Respond to the user\'s messages as a fellow improviser playing an improv game, in a way that is relevant to the context. Be a playful teammate, and don\'t hesitate to banter. And make the game fun and engaging. Your goal is to make the user feel like they are playing a game with you. And be fun, friendly, witty, and engaging. Use the provided context (location or character) to make the game relevant and interesting. Keep the responses to one or two sentences maximum'},)

        for message in request.messages:
            role = 'user' if message.isUser else 'assistant'
            messages.append({'role': role, 'content': message.text})

        response = get_completion_from_messages(messages, temperature=0.0)
        return ChatResponse(response=response)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scene_review")
async def scene_review(request: SceneReviewRequest) -> SceneReviewResponse:
    try:
        # Create context for scene review
        scene_context = ""
        if request.sceneContext.get("type") == "location":
            scene_context = f"Location: {request.sceneContext.get('value')}"
        elif request.sceneContext.get("type") == "character":
            scene_context = f"Character: {request.sceneContext.get('value')}"

        # Format the scene dialogue
        scene_dialogue = "\n".join([
            f"{'User' if msg['isUser'] else 'AI'}: {msg['text']}"
            for msg in request.messages
        ])

        messages = [
            {
                'role': 'system',
                'content': """You are an experienced improv coach and teacher tasked with analyzing and providing constructive feedback on an improv scene. Your feedback should be encouraging, supportive, and specific, while also offering valuable insights for improvement.

The improv scene is a conversation between a User and an AI improviser. Analyze the whole scene, but provide feedback only on the User's performance, and not on the AI's performance.
Before providing your final feedback, wrap your analysis in <scene_breakdown> tags to break down your thoughts and ensure a thorough review of all aspects. In your breakdown, follow these steps:

1. If appropriate, quote specific lines from the scene that demonstrate adherence to improv principles (e.g., "Yes, and...", character consistency). Explain how each quote shows good improv technique.

2. List and number 1 strong moment in the scene. Explain why it was effective.

3. If applicable, identify 1 area for improvement. Consider both performer actions and missed opportunities. Suggest a specific way the performers could have enhanced the scene.

4. Brainstorm 1-2 potential jokes, puns, or cultural references that could have been incorporated to enhance the scene. Explain how each one could have fit into the existing dialogue.

After your breakdown, provide your final feedback in bullet point format. Each bullet point should be 2-3 sentences long and cover one specific aspect of the scene. Ensure your feedback is concise, specific, and encouraging.

Your feedback should address :
• Adherence to improv principles
• Highlight of strong moments
• Suggestions for improvement
• Missed opportunities or potential enhancements

Example output structure:

<scene_breakdown>
[Your detailed analysis of the scene, covering all the points mentioned above]
</scene_breakdown>

• [Feedback on improv principles]
• [Highlight of a strong moment]
• [Suggestion for improvement]
• [Missed opportunity or potential enhancement]

Please proceed with your analysis and feedback for the given improv scene."""
            },
            {
                'role': 'user',
                'content': f"""Please review this improv scene and provide the feedback in the format specified above only for the User's dialogue. Ignore the AI's dialogue:
                \<scene_context>: {scene_context}. </scene_context>
                <scene_dialogue>
                {scene_dialogue}
                </scene_dialogue>"""
            }
        ]

        response = get_completion_from_messages(messages, temperature=0.7)
        return SceneReviewResponse(review=response)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 


def get_completion_from_messages(messages, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",  # Using the latest GPT-4 model
            messages=messages,
            temperature=temperature,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error in get_completion_from_messages:")
        print(e)
        return "Error in get_completion_from_messages"