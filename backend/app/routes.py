from flask import Blueprint, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import random

bp = Blueprint('main', __name__)

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

@bp.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the AI Dungeon Master API"}), 200

@bp.route('/generate_response', methods=['POST'])

def ai_generate_response():
    data = request.json
    user_input = data['input']
    
    # Update game state
    game_state.turns += 1
    
    # Get the most relevant context based on user input
    relevant_context = get_most_similar_context(user_input)
    
    # Generate AI response
    prompt = f"{relevant_context}\nPlayer: {user_input}\nDungeon Master:"
    ai_response = generate_response(prompt)
    
    # Update game state and history
    game_state.context = relevant_context
    game_state.history.append(f"Player: {user_input}")
    game_state.history.append(f"DM: {ai_response}")
    
    return jsonify({
        'response': ai_response,
        'game_state': {
            'turns': game_state.turns,
            'context': game_state.context,
            'history': game_state.history[-5:]  # Return last 5 exchanges
        }
    })

@bp.route('/reset', methods=['POST'])
def reset_game():
    global game_state
    game_state = GameState()
    return jsonify({
        'message': 'Game reset successfully',
        'initial_context': game_state.context
    })