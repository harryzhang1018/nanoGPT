import math
import numpy as np
# Define the reward calculation function based on your heuristic rule
def calculate_reward(text):
    # Counting colons
    colon_count = text.count(':')
    
    # Counting classical phrases
    classical_words = ['thou', 'thee', 'thy', 'hath', 'dost', 'villain', 'pray','tis','art','ye','slain','nay','lord']  # Extend this list as needed
    classical_count = sum(text.lower().count(word) for word in classical_words)
    
    # Total reward
    reward = colon_count * 5 + classical_count * 10
    return reward

# Function to load the text, divide it, and label each sample
def process_text_file(filename, num_samples):
    # Read the full text from the file
    with open(filename, 'r') as file:
        full_text = file.read()
    
    # Calculate the size of each sample
    sample_size = math.ceil(len(full_text) / num_samples)
    
    # Create labeled data for each sample
    labeled_data = []
    for i in range(num_samples):
        # Extract sample text
        start_index = i * sample_size
        end_index = min((i + 1) * sample_size, len(full_text))
        sample_text = full_text[start_index:end_index]
        
        # Calculate reward for this sample
        reward = calculate_reward(sample_text)
        
        # Append labeled data
        labeled_data.append({"sample_text": sample_text, "reward": reward})
    
    return labeled_data

# Parameters
filename = "/home/harry/grad_class/nanoGPT/reward_model/data/rawData.txt"  # Replace with your text file name
num_samples = 200  # Specify the number of samples you want

# Process the text file and print the labeled data
labeled_data = process_text_file(filename, num_samples)

import json

# Save labeled data as JSON
with open('/home/harry/grad_class/nanoGPT/reward_model/data/labeledData.json', 'w') as f:
    json.dump(labeled_data, f, indent=4)

# # Display the results
# for entry in labeled_data:
#     print(f"Reward: {entry['reward']}\n")
