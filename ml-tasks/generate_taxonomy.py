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

def create_graph(name, label):
    dot = graphviz.Digraph(name, format='png')
    dot.attr(rankdir='TB', label=label, labelloc='t', fontsize='20', fontname='Arial')
    dot.attr('node', **node_attr)
    dot.attr('edge', **edge_attr)
    return dot

def generate_supervised_taxonomy():
    dot = create_graph('supervised_tree', 'The Supervised Learning Family\n(Prediction based on Labels)')
    
    # Root
    dot.node('SL', 'Supervised Learning', fillcolor='#2E86AB', fontcolor='white', fontsize='16')
    
    # Major Branches
    dot.node('CLS', 'Classification\n(Predict Category)', fillcolor='#A9D6E5')
    dot.node('REG', 'Regression\n(Predict Number)', fillcolor='#A9D6E5')
    
    dot.edge('SL', 'CLS')
    dot.edge('SL', 'REG')
    
    # Classification Children
    dot.node('IMG_CLS', 'Image Classification\n(Is it a cat?)')
    dot.node('SENT', 'Sentiment Analysis\n(Is it positive?)')
    dot.node('SPAM', 'Spam Detection\n(Is it junk?)')
    
    dot.edge('CLS', 'IMG_CLS', label='Input: Image')
    dot.edge('CLS', 'SENT', label='Input: Text')
    dot.edge('CLS', 'SPAM', label='Input: Text')
    
    # Regression Children
    dot.node('PRICE', 'Price Prediction\n(House/Stock)')
    dot.node('AGE', 'Age Estimation\n(From Face)')
    dot.node('BB', 'Bounding Box Reg\n(Object Detection)', style='dashed')
    
    dot.edge('REG', 'PRICE')
    dot.edge('REG', 'AGE')
    dot.edge('REG', 'BB')
    
    dot.render('diagrams/taxonomy_supervised', cleanup=True)

def generate_sequence_taxonomy():
    dot = create_graph('sequence_tree', 'The Sequence Family\n(Seq2Seq & Generators)')
    dot.attr(rankdir='LR')
    
    # Root
    dot.node('SEQ', 'Sequence Models', fillcolor='#06A77D', fontcolor='white', fontsize='16')
    
    # Types
    dot.node('N21', 'Many-to-One\n(Classification)', fillcolor='#B7E4C7')
    dot.node('N2N', 'Many-to-Many\n(Seq2Seq)', fillcolor='#B7E4C7')
    dot.node('GEN', 'Generative\n(Next Token)', fillcolor='#B7E4C7')
    
    dot.edge('SEQ', 'N21')
    dot.edge('SEQ', 'N2N')
    dot.edge('SEQ', 'GEN')
    
    # Leaves
    dot.node('SENT', 'Sentiment Analysis')
    dot.edge('N21', 'SENT')
    
    dot.node('MT', 'Machine Translation\n(Text -> Text)')
    dot.node('SUM', 'Summarization\n(Long -> Short)')
    dot.node('ASR', 'Speech Recog\n(Audio -> Text)')
    
    dot.edge('N2N', 'MT')
    dot.edge('N2N', 'SUM')
    dot.edge('N2N', 'ASR')
    
    dot.node('GPT', 'Text Generation\n(GPT/LLMs)')
    dot.node('MUS', 'Music Gen')
    
    dot.edge('GEN', 'GPT')
    dot.edge('GEN', 'MUS')

    dot.render('diagrams/taxonomy_sequence', cleanup=True)

def generate_vision_taxonomy():
    dot = create_graph('vision_tree', 'Computer Vision: A Progressive Taxonomy')
    
    # Root
    dot.node('CV', 'Computer Vision', fillcolor='#9D4EDD', fontcolor='white', fontsize='16')
    
    # Levels
    dot.node('L1', 'Level 1: Whole Image\n(Classification)', shape='ellipse')
    dot.node('L2', 'Level 2: Boxes\n(Detection)', shape='ellipse')
    dot.node('L3', 'Level 3: Pixels\n(Segmentation)', shape='ellipse')
    
    dot.edge('CV', 'L1')
    dot.edge('L1', 'L2', label=' + Localization')
    dot.edge('L2', 'L3', label=' + Shape Detail')
    
    # Tasks
    dot.node('IC', 'Image Classif.')
    dot.node('OD', 'Object Detection')
    dot.node('SEM', 'Semantic Seg.\n(Classes)')
    dot.node('INS', 'Instance Seg.\n(Objects)')
    dot.node('PAN', 'Panoptic Seg.\n(Everything)')
    
    dot.edge('L1', 'IC')
    dot.edge('L2', 'OD')
    dot.edge('L3', 'SEM')
    dot.edge('L3', 'INS')
    dot.edge('INS', 'PAN', style='dashed')
    
    dot.render('diagrams/taxonomy_vision', cleanup=True)

def generate_unsupervised_taxonomy():
    dot = create_graph('unsupervised_tree', 'Unsupervised & Self-Supervised')
    
    # Root
    dot.node('US', 'No Labels?', fillcolor='#F6AE2D', fontcolor='white', fontsize='16')
    
    # Branches
    dot.node('GRP', 'Grouping\n(Clustering)', fillcolor='#FDE2E4')
    dot.node('REP', 'Representation\n(Embeddings)', fillcolor='#FDE2E4')
    dot.node('GEN', 'Synthesis\n(Generative)', fillcolor='#FDE2E4')
    
    dot.edge('US', 'GRP')
    dot.edge('US', 'REP')
    dot.edge('US', 'GEN')
    
    # Leaves
    dot.node('KM', 'K-Means')
    dot.node('LDA', 'Topic Modeling')
    dot.edge('GRP', 'KM')
    dot.edge('GRP', 'LDA')
    
    dot.node('DIM', 'Dim Reduction\n(PCA/t-SNE)')
    dot.node('SSL', 'Contrastive Learning\n(SimCLR)')
    dot.edge('REP', 'DIM')
    dot.edge('REP', 'SSL')
    
    dot.node('GAN', 'GANs/Diffusion\n(Image Gen)')
    dot.node('INP', 'Inpainting')
    dot.edge('GEN', 'GAN')
    dot.edge('GEN', 'INP')
    
    dot.render('diagrams/taxonomy_unsupervised', cleanup=True)

if __name__ == "__main__":
    generate_supervised_taxonomy()
    generate_sequence_taxonomy()
    generate_vision_taxonomy()
    generate_unsupervised_taxonomy()
