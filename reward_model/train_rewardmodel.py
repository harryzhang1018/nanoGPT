import torch
from torch import nn, optim
import json
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random

# Define the RewardModel using Transformer
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = self.transformer(**inputs)
        reward = torch.sigmoid(outputs.logits)  # Output a scalar reward
        return reward

# Define the training loop
def train_reward_model(training_data, model_name, num_epochs=10, learning_rate=0.0001):
    """
    Trains the reward model using dummy embeddings and heuristic-based rewards.

    Parameters:
    - training_data: List of tuples, each with (text, reward), where reward is a scalar reward score.
    - model_name: Name of the transformer model.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for optimizer.
    """
    
    reward_model = RewardModel(model_name=model_name).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(reward_model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Shuffle training data for each epoch
    for epoch in range(num_epochs):
        total_loss = 0
        random.shuffle(training_data)
        for text, reward in training_data:
            optimizer.zero_grad()
            
            # Forward pass to get predicted reward
            predicted_reward = reward_model([text])
            
            # Calculate loss (between predicted reward and actual reward from heuristic)
            loss = criterion(predicted_reward, torch.tensor([[reward]], dtype=torch.float32).to(device))
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    return reward_model

# Example usage
model_name = 'distilbert-base-uncased'  # Define the transformer model name
num_epochs = 50  # Increase the number of epochs to allow better learning
learning_rate = 0.00005  # Reduce the learning rate to improve convergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Load the json file
data_file_path = '/home/harry/grad_class/nanoGPT/reward_model/data/labeledData.json'
with open(data_file_path, 'r') as f:
    data = json.load(f)

# Extract texts and rewards from the JSON data
texts = [item['sample_text'] for item in data]
rewards = [item['reward'] for item in data]

# Normalize rewards to be in range [0, 1]
max_reward = max(rewards)
min_reward = min(rewards)
print(f"Max reward: {max_reward}, Min reward: {min_reward}")
rewards = [(r - min_reward) / (max_reward - min_reward) for r in rewards]

# Prepare the training data for your model
training_data = list(zip(texts, rewards))

# Train the reward model
trained_reward_model = train_reward_model(
    training_data=training_data,
    model_name=model_name,
    num_epochs=num_epochs,
    learning_rate=learning_rate
)

# Export the trained model
def save_reward_model(reward_model, path):
    """
    Saves the trained reward model to the specified path.
    
    Parameters:
    - reward_model: Trained reward model.
    - path: Path to save the model.
    """
    torch.save(reward_model.state_dict(), path)
    print(f"Model saved to {path}")

# Save the trained reward model
save_path = '/home/harry/grad_class/nanoGPT/reward_model/trained_reward_model.pth'
save_reward_model(trained_reward_model, save_path)

# Step 2: Define the function to get the score for an input sentence
def get_reward_score(sentence, reward_model):
    # Pass the sentence through the reward model to get the score
    with torch.no_grad():  # No need to calculate gradients for inference
        score = reward_model([sentence]).item()  # Convert the single output to a scalar
    
    return score


# Example usage
shakespeare_sentence = "BENVOLIO:\nAy, we take her better for this fool.\nROMEO:\nHere comes me, get him here no more.\nNurse:\nWhy shall we speak? what, without off this?"
sscore = get_reward_score(shakespeare_sentence, trained_reward_model)
print(f"Reward score for the Shakespeare sentence: {sscore*(max_reward - min_reward) + min_reward}")

normal_sentence = "Welcome to ChatGPT, an advanced language model developed by OpenAI. This conversational AI can assist you in a wide range of tasks—answering questions, brainstorming ideas, providing recommendations, helping with coding, and much more."
nscore = get_reward_score(normal_sentence, trained_reward_model)
print(f"Reward score for the normal sentence: {nscore*(max_reward - min_reward) + min_reward}")
