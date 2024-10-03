from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import random

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined contexts and responses
contexts = [
    "You are in a dark forest. The trees loom over you, their branches reaching out like gnarled fingers.",
    "You stand before an ancient temple. Its stone walls are covered in mysterious symbols.",
    "You find yourself in a bustling medieval marketplace. The air is filled with the scents of spices and the sounds of haggling.",
    "You're at the entrance of a deep cave. A cool breeze emanates from its depths.",
    "You're on a ship sailing through stormy seas. The deck rocks beneath your feet."
]

class GameState:
    def __init__(self):
        self.turns = 0
        self.context = random.choice(contexts)
        self.history = [self.context]

game_state = GameState()

class UserInput(BaseModel):
    input: str

def generate_response(prompt, max_length=100):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = gpt2_tokenizer.eos_token_id
    
    output = gpt2_model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def get_most_similar_context(query):
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    context_embeddings = sentence_model.encode(contexts, convert_to_tensor=True)
    
    cosine_scores = util.pytorch_cos_sim(query_embedding, context_embeddings)
    best_match = torch.argmax(cosine_scores)
    
    return contexts[best_match.item()]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Dungeon Master API"}

@app.post("/generate_response")
async def ai_generate_response(user_input: UserInput):
    global game_state
    
    # Update game state
    game_state.turns += 1
    
    # Get the most relevant context based on user input
    relevant_context = get_most_similar_context(user_input.input)
    
    # Generate AI response
    prompt = f"{relevant_context}\nPlayer: {user_input.input}\nDungeon Master:"
    ai_response = generate_response(prompt)
    
    # Update game state and history
    game_state.context = relevant_context
    game_state.history.append(f"Player: {user_input.input}")
    game_state.history.append(f"DM: {ai_response}")
    
    return {
        'response': ai_response,
        'game_state': {
            'turns': game_state.turns,
            'context': game_state.context,
            'history': game_state.history[-5:]  # Return last 5 exchanges
        }
    }

@app.post("/reset")
async def reset_game():
    global game_state
    game_state = GameState()
    return {
        'message': 'Game reset successfully',
        'initial_context': game_state.context
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)