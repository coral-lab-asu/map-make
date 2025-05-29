import json
import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.cuda.empty_cache()

# Load pre-trained model and tokenizer
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU
model.to(device)

# Batched NLI function using RoBERTa
def nli_roberta_batch(pairs):
    # Unzip the sentence pairs into two lists of sentences
    sentences1, sentences2 = zip(*pairs)

    # Tokenize the input sentences and prepare them for the model
    inputs = tokenizer(list(sentences1), list(sentences2), return_tensors="pt", truncation=True, padding=True)

    # Move input tensors to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities of the classes (entailment, neutral, contradiction)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().tolist()  # Move back to CPU for easier processing

    # Return entailment probabilities (index 0 in the probabilities)
    entailment_scores = [prob[-1] for prob in probabilities]
    
    return entailment_scores

# Precision, Recall, and F1 calculation based on the formulas
def precision_recall_f1(predicted_statements, ground_truth_statements):
    N = len(predicted_statements)  # Number of predicted statements
    M = len(ground_truth_statements)  # Number of ground truth statements

    # Create all combinations of predicted and ground truth statements
    predicted_to_ground_truth_pairs = [(pi, gj) for pi in predicted_statements for gj in ground_truth_statements]
    ground_truth_to_predicted_pairs = [(gj, pi) for gj in ground_truth_statements for pi in predicted_statements]

    # Precision: Max entailment for each predicted statement over all ground truth statements
    precision_entailment_scores = nli_roberta_batch(predicted_to_ground_truth_pairs)  # Batch NLI scores
    precision_scores = []
    for i in range(N):
        # Each predicted statement compares to M ground truth statements, so take max over M
        max_score = max(precision_entailment_scores[i * M:(i + 1) * M])
        precision_scores.append(max_score)
    precision = sum(precision_scores) / N

    # Recall: Max entailment for each ground truth statement over all predicted statements
    recall_entailment_scores = nli_roberta_batch(ground_truth_to_predicted_pairs)  # Batch NLI scores
    recall_scores = []
    for j in range(M):
        # Each ground truth statement compares to N predicted statements, so take max over N
        max_score = max(recall_entailment_scores[j * N:(j + 1) * N])
        recall_scores.append(max_score)
    recall = sum(recall_scores) / M

    # F1 Score: Harmonic mean of precision and recall
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score


def parse_atomic_statements(text):
    if text == '':
        return ''
    # Initialize an empty list to store statements
    statements = []
    
    # Split the input text into individual lines
    lines = text.splitlines()
    
    # Flag to indicate if we are in the 'Statements' section
    in_statements = False
    
    for line in lines:
        # Check if the current line marks the start of 'Statements'
        if line.strip().startswith('Statements:'):
            in_statements = True
            continue  # Move to the next line
        
        # If we are in the 'Statements' section
        if in_statements:
            # If we encounter an empty line or another section, stop processing
            if not line.strip() or line.strip().startswith('Rows:'):
                break
            
            # Use regex to match lines that start with a number and a period
            match = re.match(r'\d+\.\s*(.*)', line)
            if match:
                # Extract the statement without the numbering
                statement = match.group(1).strip()
                statements.append(statement)
    
    return statements


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', 
        type=str,
        required=True,
        help='Input tabunrolled path'
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        required=True,
        help='Output entailment path'
    )
    parser.add_argument(
        '--gold_path',
        type=str,
        required=True,
        help='Gold table tabunrolled path'
    )
    args = parser.parse_args()

    method_path = args.input_path
    eval_path = args.output_path

    gold_player_path = os.path.join(args.gold_path, 'player/')
    gold_team_path = os.path.join(args.gold_path, 'team/')

    error_idx = []
    adobe_eval_output = {}
    for idx in range(728):
        try:
            print('Idx: ', idx)
            
            with open(os.path.join(gold_team_path, f'{idx}.txt'), 'r') as f:
                gold_team = parse_atomic_statements(f.read())

            with open(os.path.join(gold_player_path, f'{idx}.txt'), 'r') as f:
                gold_player = parse_atomic_statements(f.read())

            with open(os.path.join(method_path, 'team', f'{idx}.txt'), 'r') as f:
                method_team = parse_atomic_statements(f.read())

            with open(os.path.join(method_path, 'player', f'{idx}.txt'), 'r') as f:
                method_player = parse_atomic_statements(f.read())
            

            if gold_team == '' or method_team == '' or len(gold_team) == 0 or len(method_team) == 0:
                tp, tr, tf1 = 0, 0, 0
            else:
                tp, tr, tf1 = precision_recall_f1(method_team, gold_team)
            if gold_player == '' or method_player == '' or len(gold_player) == 0 or len(method_player) == 0:
                pp, pr, pf1 = 0, 0, 0
            else:
                pp, pr, pf1 = precision_recall_f1(method_player, gold_player)
            
            adobe_eval_output[idx] = {}
            adobe_eval_output[idx]['Team'] = {'Precision': tp, 'Recall': tr, 'F1-score': tf1}
            adobe_eval_output[idx]['Player'] = {'Precision': pp, 'Recall': pr, 'F1-score': pf1}

        except Exception as e:
            print('Idx: ', idx)
            print(e)
            error_idx.append(int(idx))

    with open(eval_path, 'w') as f:
        json.dump(adobe_eval_output, f, indent=6)

    print(error_idx)
    print(len(error_idx))
    