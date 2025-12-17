import graphviz
import os

os.makedirs('diagrams', exist_ok=True)

# Common styling
node_attr = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': '#f8f9fa',
    'color': '#2E86AB',
    'fontname': 'Arial',
    'fontsize': '12'
}

edge_attr = {
    'color': '#6c757d',
    'arrowhead': 'vee'
}

def create_graph(name, label, rankdir='LR'):
    dot = graphviz.Digraph(name, format='png')
    dot.attr(rankdir=rankdir, label=label, labelloc='t', fontsize='20', fontname='Arial', compound='true')
    dot.attr('node', **node_attr)
    dot.attr('edge', **edge_attr)
    return dot

# 1. NTP Problem Statement (Graphviz version)
def plot_ntp_problem_gv():
    dot = create_graph('ntp_problem_gv', 'Next Token Prediction: The Core Problem', rankdir='LR')
    
    dot.node('Input', 'Given: "app"', fillcolor='#A9D6E5')
    dot.node('Model', 'Predictive Model', fillcolor='#B7E4C7')
    dot.node('Output', 'Predict: "?" (next character)', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Input', 'Model')
    dot.edge('Model', 'Output')
    
    dot.render('diagrams/ntp_problem_gv', cleanup=True)

# 2. Sliding Window (Graphviz version - showing process)
def plot_sliding_window_gv():
    dot = create_graph('ntp_sliding_window_gv', 'Creating Training Data: Sliding Window', rankdir='TB')
    
    dot.node('Word', 'Source Word: "aabid"', fillcolor='#A9D6E5')
    
    with dot.subgraph(name='cluster_examples') as c:
        c.attr(style='filled', color='#FDE2E4', label='Training Examples')
        c.node('Ex1', 'Context: "..." -> Target: "a"')
        c.node('Ex2', 'Context: "..a" -> Target: "a"')
        c.node('Ex3', 'Context: ".aa" -> Target: "b"')
        c.node('Ex4', 'Context: "aab" -> Target: "i"')
        c.node('Ex5', 'Context: "abi" -> Target: "d"')
        c.edge('Ex1', 'Ex2', style='invis') # For vertical alignment
        c.edge('Ex2', 'Ex3', style='invis')
        c.edge('Ex3', 'Ex4', style='invis')
        c.edge('Ex4', 'Ex5', style='invis')

    dot.edge('Word', 'Ex1', label='Slide Window')
    
    dot.render('diagrams/ntp_sliding_window_gv', cleanup=True)

# 3. Architecture (MLP) - Graphviz version
def plot_architecture_gv():
    dot = create_graph('ntp_architecture_gv', 'MLP Architecture for Next Character Prediction', rankdir='LR')
    
    dot.node('InputChars', 'Input Characters\n(e.g., "a", "a", "b")', fillcolor='#A9D6E5')
    dot.node('Embed', 'Lookup Embeddings\n(Convert to Vectors)', fillcolor='#B7E4C7')
    dot.node('Concat', 'Concatenate Embeddings\n(Form Context Vector)', fillcolor='#FDE2E4')
    dot.node('Hidden', 'Hidden Layer (MLP)\n(Extract Features)', fillcolor='#A9D6E5')
    dot.node('OutputLogits', 'Output Logits\n(Scores for 27 chars)', fillcolor='#B7E4C7')
    dot.node('Softmax', 'Softmax\n(Probabilities)', fillcolor='#FDE2E4')
    dot.node('Prediction', 'Next Character Prediction\n(Prob. Distribution)', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('InputChars', 'Embed')
    dot.edge('Embed', 'Concat')
    dot.edge('Concat', 'Hidden')
    dot.edge('Hidden', 'OutputLogits')
    dot.edge('OutputLogits', 'Softmax')
    dot.edge('Softmax', 'Prediction')
    
    dot.render('diagrams/ntp_architecture_gv', cleanup=True)

# 4. Sampling Tree (Graphviz version)
def plot_sampling_tree_gv():
    dot = create_graph('ntp_sampling_tree_gv', 'Auto-Regressive Generation (Sampling Tree)', rankdir='LR')
    
    dot.node('Root', 'Start: "..."', fillcolor='#A9D6E5')
    
    dot.node('L1_a', '"...a"\n(P=0.4)', fillcolor='#B7E4C7')
    dot.node('L1_b', '"...b"\n(P=0.2)', fillcolor='#B7E4C7')
    dot.node('L1_c', '"...c"\n(P=0.1)', fillcolor='#B7E4C7')
    
    dot.node('L2_aa', '"...aa"\n(P=0.1)')
    dot.node('L2_ab', '"...ab"\n(P=0.3)')
    dot.node('L2_bi', '"...bi"\n(P=0.6)')
    
    dot.node('Final_aab', '"...aab"', fillcolor='#FDE2E4')
    dot.node('Final_abid', '"...abid"', fillcolor='#2E86AB', fontcolor='white')

    dot.edge('Root', 'L1_a', label='Sample 'a'')
    dot.edge('Root', 'L1_b', label='Sample 'b'')
    dot.edge('Root', 'L1_c', label='Sample 'c'')
    
    dot.edge('L1_a', 'L2_aa')
    dot.edge('L1_a', 'L2_ab')
    dot.edge('L1_b', 'L2_bi')
    
    dot.edge('L2_ab', 'Final_aab')
    dot.edge('L2_bi', 'Final_abid')
    
    dot.render('diagrams/ntp_sampling_tree_gv', cleanup=True)

# 5. Tokenization Visual (Graphviz - showing hierarchy)
def plot_tokenization_gv():
    dot = create_graph('ntp_tokenization_gv', 'Tokenization Levels', rankdir='TB')
    
    dot.node('Text', 'Original Text: "cooking"', fillcolor='#A9D6E5')
    
    dot.node('Char', 'Character-Level\n(c-o-o-k-i-n-g)', fillcolor='#B7E4C7')
    dot.node('Word', 'Word-Level\n(cooking)', fillcolor='#B7E4C7')
    dot.node('Subword', 'Subword/Token-Level\n(cook-ing)', fillcolor='#FDE2E4')
    
    dot.node('CharPros', 'Pros: Small Vocab, Handles OOV\nCons: Long Sequences', shape='note')
    dot.node('WordPros', 'Pros: Shorter Sequences\nCons: Huge Vocab, OOV', shape='note')
    dot.node('SubwordPros', 'Pros: Balanced Vocab, Good for LLMs\nCons: More complex', shape='note')
    
    dot.edge('Text', 'Char')
    dot.edge('Text', 'Word')
    dot.edge('Text', 'Subword')
    
    dot.edge('Char', 'CharPros')
    dot.edge('Word', 'WordPros')
    dot.edge('Subword', 'SubwordPros')

    dot.render('diagrams/ntp_tokenization_gv', cleanup=True)

# 6. One-Hot vs. Dense (Graphviz version)
def plot_onehot_gv():
    dot = create_graph('ntp_onehot_gv', 'One-Hot vs. Dense Embeddings', rankdir='LR')
    
    dot.node('Char', 'Character (e.g., \'a\')', fillcolor='#A9D6E5')
    
    dot.node('OneHot', 'One-Hot Vector\n[0,0,1,0,...]\n(Sparse, No Meaning)', fillcolor='#B7E4C7')
    dot.node('Dense', 'Dense Embedding Vector\n[0.2, -0.5, 1.3]\n(Compact, Captures Meaning)', fillcolor='#FDE2E4')
    
    dot.edge('Char', 'OneHot', label='Traditional')
    dot.edge('Char', 'Dense', label='Modern ML')
    
    dot.render('diagrams/ntp_onehot_gv', cleanup=True)

# 7. Context Cutoff (Graphviz version - conceptual)
def plot_context_cutoff_gv():
    dot = create_graph('ntp_context_cutoff_gv', 'The Context Cutoff Problem', rankdir='TB')
    
    dot.node('Sentence', 'Long Sentence (e.g., Alice and the key...)', fillcolor='#A9D6E5')
    dot.node('Window', 'Fixed Context Window\n("door with the ___"', fillcolor='#FDE2E4')
    dot.node('LostInfo', 'Key Information Lost!\n("key" not in window)', fillcolor='#E63946', fontcolor='white')
    dot.node('BadPred', 'Bad Prediction', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Sentence', 'Window')
    dot.edge('Window', 'LostInfo', label='Limited View')
    dot.edge('LostInfo', 'BadPred')
    
    dot.render('diagrams/ntp_context_cutoff_gv', cleanup=True)

# 8. RNN Concept (Graphviz version)
def plot_rnn_gv():
    dot = create_graph('ntp_rnn_gv', 'Recurrent Neural Network (RNN) Concept', rankdir='LR')
    
    dot.node('Input1', 'Word 1', fillcolor='#A9D6E5')
    dot.node('H1', 'Hidden State (h1)', shape='circle', fillcolor='#B7E4C7')
    dot.node('Output1', 'Output 1')
    
    dot.node('Input2', 'Word 2', fillcolor='#A9D6E5')
    dot.node('H2', 'Hidden State (h2)', shape='circle', fillcolor='#B7E4C7')
    dot.node('Output2', 'Output 2')
    
    dot.node('InputN', 'Word N', fillcolor='#A9D6E5')
    dot.node('HN', 'Hidden State (hN)', shape='circle', fillcolor='#B7E4C7')
    dot.node('OutputN', 'Output N')
    
    dot.edge('Input1', 'H1')
    dot.edge('H1', 'Output1')
    dot.edge('H1', 'H2', label='Passes Memory')
    
    dot.edge('Input2', 'H2')
    dot.edge('H2', 'Output2')
    dot.edge('H2', 'HN', label='...')
    
    dot.edge('InputN', 'HN')
    dot.edge('HN', 'OutputN')
    
    dot.render('diagrams/ntp_rnn_gv', cleanup=True)

# 9. QKV Analogy (Graphviz version)
def plot_qkv_gv():
    dot = create_graph('ntp_qkv_gv', 'Self-Attention: The QKV Mechanism Analogy', rankdir='LR')
    
    dot.node('Query', 'Query (Q)\n(What am I looking for?)', fillcolor='#A9D6E5')
    
    dot.node('Keys', 'Keys (K)\n(What do I contain?)', fillcolor='#B7E4C7')
    dot.node('Values', 'Values (V)\n(What is my content?)', fillcolor='#FDE2E4')
    
    dot.node('Match', 'Similarity Score\n(Query vs. Keys)', shape='box')
    dot.node('WeightedSum', 'Weighted Sum of Values\n(Based on Scores)', shape='box')
    dot.node('Output', 'Attention Output\n(Contextualized Value)', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Query', 'Match', label='Compare')
    dot.edge('Keys', 'Match', label='Compare')
    dot.edge('Match', 'WeightedSum', label='Weights')
    dot.edge('Values', 'WeightedSum', label='Content')
    dot.edge('WeightedSum', 'Output')
    
    dot.render('diagrams/ntp_qkv_gv', cleanup=True)

# 10. Attention Concept (Graphviz version)
def plot_attention_concept_gv():
    dot = create_graph('ntp_attention_concept_gv', 'Attention: Focusing on Relevant Context', rankdir='TB')
    
    dot.node('CurrentWord', 'Current Word: "it"', fillcolor='#F6AE2D')
    
    with dot.subgraph(name='cluster_context') as c:
        c.attr(style='filled', color='#A9D6E5', label='Context Words')
        c.node('Animal', '"animal"', fillcolor='#B7E4C7')
        c.node('Didnt', '"didn\'t"', fillcolor='#B7E4C7')
        c.node('Cross', '"cross"', fillcolor='#B7E4C7')
        c.node('Street', '"street"', fillcolor='#B7E4C7')
        c.node('Tired', '"tired"', fillcolor='#B7E4C7')
        c.edge('Animal', 'Didnt', style='invis')
        c.edge('Didnt', 'Cross', style='invis')
        c.edge('Cross', 'Street', style='invis')
        c.edge('Street', 'Tired', style='invis')

    dot.edge('CurrentWord', 'Animal', label='High Attention Score\n("it" -> "animal")', color='#E63946', penwidth='2')
    dot.edge('CurrentWord', 'Tired', label='Medium Attention Score\n("it" -> "tired")', color='#F6AE2D')
    dot.edge('CurrentWord', 'Street', label='Low Attention Score')

    dot.render('diagrams/ntp_attention_concept_gv', cleanup=True)

if __name__ == "__main__":
    plot_ntp_problem_gv()
    plot_sliding_window_gv()
    plot_architecture_gv()
    plot_sampling_tree_gv()
    plot_tokenization_gv()
    plot_onehot_gv()
    plot_context_cutoff_gv()
    plot_rnn_gv()
    plot_qkv_gv()
    plot_attention_concept_gv()
