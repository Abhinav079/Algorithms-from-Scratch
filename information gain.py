import math
from collections import Counter

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
# Convert the data to a pandas DataFrame

def entropy(data):
    labels = data['PlayTennis']
    total_instances = len(labels)
    label_counts = Counter(labels)
    
    entropy_value = 0
    for count in label_counts.values():
        probability = count / total_instances
        entropy_value -= probability * math.log2(probability)
    
    return entropy_value
def information_gain(data, attribute):
    total_instances = len(data['PlayTennis'])
    values = set(data[attribute])
    
    gain = entropy(data)
    for value in values:
        subset = {key: [val for i, val in enumerate(data[key]) if data[attribute][i] == value] for key in data}
        subset_entropy = entropy(subset)
        subset_instances = len(subset['PlayTennis'])
        gain -= (subset_instances / total_instances) * subset_entropy
    
    return gain

def find_best_attribute(data, attributes):
    gains = {attribute: information_gain(data, attribute) for attribute in attributes}
    best_attribute = max(gains, key=gains.get)
    return best_attribute

def build_decision_tree(data, attributes):
    labels = data['PlayTennis']
    
    # If all instances have the same label, return a leaf node with that label
    if len(set(labels)) == 1:
        return labels[0]
    
    # If no attributes left, return the majority label
    if not attributes:
        majority_label = max(Counter(labels), key=Counter(labels).get)
        return majority_label
    
    # Find the best attribute to split on
    best_attribute = find_best_attribute(data, attributes)
    
    # Create a tree node with the best attribute
    tree = {best_attribute: {}}
    
    # Remove the best attribute from the list of attributes
    new_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Recursively build subtrees for each value of the best attribute
    for value in set(data[best_attribute]):
        subset = {key: [val for i, val in enumerate(data[key]) if data[best_attribute][i] == value] for key in data}
        subtree = build_decision_tree(subset, new_attributes)
        tree[best_attribute][value] = subtree
    
    return tree

# Example usage
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
decision_tree = build_decision_tree(data, attributes)
print(decision_tree)
