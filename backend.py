from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uvicorn
import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)


INTENT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CREATIVE_MODEL = "llama-3.3-70b-versatile"
CF_MODEL_ID = "@cf/black-forest-labs/flux-1-schnell"


app = FastAPI(
    title="AI Review SaaS API",
    description="Structured Instruction-Based Groq + Cloudflare Pipeline",
    version="7.0.0"
)

class ReviewRequest(BaseModel):
    review: str
    author: str
    style: str


def groq_json_chat(model: str, system_prompt: str, user_input: str,
                   temperature=0.3, max_tokens=800):

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Error: {str(e)}")


def classify_nature_of_input(user_input: str):

    system_prompt = """
You are an elite behavioral psychology strategist and consumer persuasion analyst.

Your task is NOT to summarize the review.
Your task is to reverse-engineer the psychological motive behind it.

You must analyze at three levels:

LEVEL 1 — Surface Layer
- What the user explicitly says
- Observable sentiment (positive, mixed, defensive, aspirational, etc.)

LEVEL 2 — Emotional Layer
- The underlying emotional state
- Validation seeking, status signaling, justification, regret reduction,
  cognitive dissonance, pride, relief, insecurity, belonging, authority, etc.

LEVEL 3 — Strategic Layer
- Why the user chose to write this review
- What outcome they subconsciously want
- What social or psychological reinforcement they are seeking
- What persuasion angle would amplify this message in marketing

You must infer intention even if not directly stated.
Do not repeat the review.
Do not paraphrase the review.

Return STRICT JSON in this exact format:

{
  "core_intent": "The fundamental psychological reason the review exists.",
  "goal_of_user": "What outcome the user wants (social proof, validation, influence, justification, authority, etc.).",
  "emotional_state": "Dominant emotional tone beneath the surface language.",
  "psychological_driver": "Deep behavioral mechanism driving the expression (status reinforcement, risk reduction, belonging, identity alignment, etc.).",
  "marketing_angle": "How a brand should strategically position this message for maximum persuasion.",
  "execution_instruction": "A clear, direct command telling the creative system how to frame the image psychologically."
}

CRITICAL RULES:

- Be analytical, not emotional.
- Do not moralize.
- Do not generalize vaguely.
- Avoid generic phrases like 'the user is happy.'
- Be precise and inferential.
- execution_instruction must be written as a command.
  Example structure:
  'Position the scene to emphasize professional transformation and identity elevation through subtle environmental cues.'

Only return valid JSON.
No markdown.
No commentary.
No explanation outside JSON.
"""

    return groq_json_chat(
        INTENT_MODEL,
        system_prompt,
        user_input,
        temperature=0.1,
        max_tokens=600
    )



def get_creative_idea(intent_json: dict,
                      original_review: str,
                      author: str,
                      style: str):

    system_prompt = """
You are a world-class senior advertising creative director 
with deep expertise in behavioral psychology, luxury branding, 
visual storytelling, and high-conversion social media marketing.

You are NOT allowed to ignore the execution_instruction field.
It is a strategic directive and must guide:

- Emotional framing
- Composition
- Visual hierarchy
- Lighting mood
- Scene context
- Subtle persuasive cues

You are designing a high-performing Instagram marketing post
that must feel authentic, real, and emotionally persuasive.

CRITICAL RULES:

1. The image must look like a real-life professional photograph.
2. Use real-world materials (paper, desk, glass, fabric, concrete, wood, metal, etc.).
3. Lighting must be natural (window light, soft indoor ambient, golden hour, studio realism).
4. Include realistic depth of field and subtle shadows.
5. Avoid:
   - Digital art look
   - CGI appearance
   - Unrealistic glow
   - Fantasy effects
   - Hyper-saturation
   - Over-stylized typography
6. The design must feel believable and physically possible to photograph.

You must psychologically align the image with:
- The emotional_state
- The psychological_driver
- The marketing_angle

The goal is conversion-driven emotional resonance,
not aesthetic experimentation.

Return STRICT JSON in this format:

{
  "creative_hook": "A short compelling marketing hook aligned with psychological intent.",
  "image_prompt": "A detailed, photorealistic image generation prompt describing scene, materials, lighting, mood, camera perspective, environment, and placement of text if applicable.",
  "visual_direction": "A professional explanation of composition, framing, depth, lighting style, and emotional tone."
}

Do not include explanations outside JSON.
Do not include markdown.
Do not include commentary.
Only valid JSON.
"""
    user_context = {
        "psychological_strategy": intent_json,
        "review": original_review,
        "author": author,
        "style": style
    }

    return groq_json_chat(
        CREATIVE_MODEL,
        system_prompt,
        json.dumps(user_context),
        temperature=0.85,
        max_tokens=1000
    )


@app.get("/")
async def health():
    return {"status": "online"}


@app.post("/generate")
async def generate(data: ReviewRequest):

    # Step 1: Psychological Intent → Instruction
    intent = classify_nature_of_input(data.review)

    # Step 2: Creative follows instruction
    creative = get_creative_idea(
        intent,
        data.review,
        data.author,
        data.style
    )

    final_prompt = creative["image_prompt"]

    # Step 3: Cloudflare Image Generation
    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL_ID}"

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": final_prompt,
        "width": 1080,
        "height": 1080,
        "steps": 4
    }

    response = requests.post(cf_url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    result = response.json()
    img_b64 = result.get("result", {}).get("image") or result.get("image")

    if not img_b64:
        raise HTTPException(status_code=500, detail="Image generation failed")

    return {
        "success": True,
        "intent_analysis": intent,
        "creative_strategy": creative,
        "image_b64": img_b64,
        "llm_models": {
            "intent_model": INTENT_MODEL,
            "creative_model": CREATIVE_MODEL
        },
        "image_model": CF_MODEL_ID
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)