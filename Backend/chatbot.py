# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')

# 1. Load and preprocess data
print("Loading dataset...")
df = pd.read_csv('Dataset.csv')
print(f"Dataset shape: {df.shape}")
print("\nUnique emotions:")
print(df['emotion'].value_counts())

# 2. Create Dataset Class
class EmpatheticDataset(Dataset):
    def __init__(self, data, max_length=100):
        self.data = data
        self.max_length = max_length
        self.le = LabelEncoder()
        
        # Encode emotions
        self.data['emotion_encoded'] = self.le.fit_transform(self.data['emotion'])
        self.num_emotions = len(self.le.classes_)
        
        # Build vocabulary
        self.build_vocab()
        
    def build_vocab(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for text in self.data['situation'].fillna(''):
            tokens = word_tokenize(str(text).lower())
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1
        
        self.vocab_size = len(self.word2idx)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        situation = self.data.iloc[idx]['situation']
        emotion = self.data.iloc[idx]['emotion_encoded']
        
        # Tokenize and pad
        tokens = word_tokenize(str(situation).lower())
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        if len(indices) < self.max_length:
            indices += [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return {
            'text': torch.tensor(indices),
            'emotion': torch.tensor(emotion, dtype=torch.long)
        }

# 3. Create Model Class
class EmpatheticLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_emotions, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc1(hidden))
        return self.fc2(hidden)

# 4. Training Configuration
config = {
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'max_length': 100,
    'confidence_threshold': 0.7
}

# 5. Prepare Data
print("\nPreparing datasets...")
# Create full dataset first to get vocab and label encoder
full_dataset = EmpatheticDataset(df, max_length=config['max_length'])

# Split data
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Create train and validation datasets using the same word2idx and label encoder
class EmpatheticDatasetSplit(Dataset):
    def __init__(self, data, word2idx, le, max_length=100):
        self.data = data
        self.word2idx = word2idx
        self.le = le
        self.max_length = max_length
        self.data['emotion_encoded'] = self.le.transform(self.data['emotion'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return full_dataset.__getitem__(idx)

train_dataset = EmpatheticDatasetSplit(train_data, full_dataset.word2idx, full_dataset.le, config['max_length'])
val_dataset = EmpatheticDatasetSplit(val_data, full_dataset.word2idx, full_dataset.le, config['max_length'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

# 6. Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = EmpatheticLSTM(
    vocab_size=full_dataset.vocab_size,
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_emotions=full_dataset.num_emotions,
    num_layers=config['num_layers']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# 7. Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            texts = batch['text'].to(device)
            emotions = batch['emotion'].to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, emotions)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text'].to(device)
                emotions = batch['emotion'].to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, emotions)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += emotions.size(0)
                correct += (predicted == emotions).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%\n')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

# 8. Train Model
print("\nStarting training...")
train_losses, val_losses = train_model(
    train_loader, val_loader, model, criterion, optimizer, 
    config['num_epochs'], device
)

# 9. Save necessary files
print("\nSaving model and configuration files...")
torch.save(model.state_dict(), 'model.pth')
with open('word2idx.pkl', 'wb') as f:
    pickle.dump(full_dataset.word2idx, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(full_dataset.le, f)
with open('config.json', 'w') as f:
    json.dump(config, f)

# 10. Create Chatbot Class for Testing
class EmpatheticChatbot:
    def __init__(self, model_path='model.pth', config_path='config.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer and label encoder
        with open('word2idx.pkl', 'rb') as f:
            self.word2idx = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)
            
        # Load responses
        with open('responses.json', 'r') as f:
            self.responses = json.load(f)
            
        # Initialize model
        self.model = EmpatheticLSTM(
            vocab_size=len(self.word2idx),
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_emotions=len(self.le.classes_),
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def get_response(self, text):
        # Tokenize and convert to tensor
        tokens = word_tokenize(str(text).lower())
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        if len(indices) < self.config['max_length']:
            indices += [self.word2idx['<PAD>']] * (self.config['max_length'] - len(indices))
        else:
            indices = indices[:self.config['max_length']]
            
        input_tensor = torch.tensor(indices).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            predicted_emotion = self.le.inverse_transform([prediction.item()])[0]
            
            # Get response based on emotion and confidence
            if confidence.item() < self.config['confidence_threshold']:
                return {
                    'response': self.responses['low_confidence'][0],
                    'emotion': predicted_emotion,
                    'confidence': confidence.item()
                }
            
            import random
            response = random.choice(self.responses[predicted_emotion])
            
            return {
                'response': response,
                'emotion': predicted_emotion,
                'confidence': confidence.item()
            }

# 11. Test the Chatbot
print("\nInitializing chatbot for testing...")
chatbot = EmpatheticChatbot()

# Test with some example inputs
test_inputs = [
    "I'm feeling really happy today!",
    "I'm so stressed about my upcoming exam",
    "I feel really lonely and sad",
    "I just got promoted at work!"
]

print("\nTesting chatbot with example inputs:")
for input_text in test_inputs:
    response = chatbot.get_response(input_text)
    print(f"\nUser: {input_text}")
    print(f"Detected Emotion: {response['emotion']} (Confidence: {response['confidence']:.2f})")
    print(f"Bot: {response['response']}")

print("\nSetup complete! You can now interact with the chatbot using chatbot.get_response()")
