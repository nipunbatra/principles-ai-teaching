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

# 1. Detection Definition (Simplified for Graphviz)
def plot_detection_definition():
    dot = create_graph('detection_definition', 'Object Detection: What & Where')
    
    dot.node('Input', 'Input Image', fillcolor='#A9D6E5')
    dot.node('CLS', 'Classification\n(What is it?)', fillcolor='#B7E4C7')
    dot.node('LOC', 'Localization\n(Where is it?)', fillcolor='#FDE2E4')
    dot.node('Output', 'Output: Class + Box', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Input', 'CLS')
    dot.edge('Input', 'LOC')
    dot.edge('CLS', 'Output')
    dot.edge('LOC', 'Output')
    
    dot.render('diagrams/detection_definition', cleanup=True)

# 2. NMS (Non-Maximum Suppression) Flowchart
def plot_nms_flow():
    dot = create_graph('nms_flow', 'Non-Maximum Suppression (NMS) Flow', rankdir='TB')
    
    dot.node('Start', 'Detector Output (Many Boxes)', shape='oval', fillcolor='#A9D6E5')
    dot.node('Threshold', 'Discard Low Confidence Boxes\n(e.g., score < 0.5)', shape='box')
    dot.node('LoopStart', 'While there are remaining boxes', shape='diamond', fillcolor='#FDE2E4')
    dot.node('PickBest', 'Pick Box with Highest Confidence', shape='box')
    dot.node('AddToFinal', 'Add to Final Detections', shape='box')
    dot.node('OverlapCheck', 'Discard Overlapping Boxes\n(IoU > Threshold)', shape='box')
    dot.node('End', 'Final Detections', shape='oval', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Start', 'Threshold')
    dot.edge('Threshold', 'LoopStart')
    dot.edge('LoopStart', 'PickBest', label='Yes')
    dot.edge('PickBest', 'AddToFinal')
    dot.edge('AddToFinal', 'OverlapCheck')
    dot.edge('OverlapCheck', 'LoopStart', label='Continue')
    dot.edge('LoopStart', 'End', label='No')
    
    dot.render('diagrams/nms_flow', cleanup=True)

# 3. Generic Detection Pipeline
def plot_generic_pipeline():
    dot = create_graph('generic_pipeline', 'Generic Object Detection Pipeline', rankdir='LR')
    
    dot.node('Input', 'Input Image', fillcolor='#A9D6E5')
    dot.node('Backbone', 'Backbone CNN\n(e.g., ResNet)', fillcolor='#B7E4C7')
    dot.node('Features', 'Feature Maps', shape='folder')
    dot.node('Head', 'Detection Head\n(Classifier + Regressor)', fillcolor='#FDE2E4')
    dot.node('Output', 'Bounding Boxes + Classes', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Input', 'Backbone', label='Extract Features')
    dot.edge('Backbone', 'Features')
    dot.edge('Features', 'Head', label='Predict BBoxes & Classes')
    dot.edge('Head', 'Output')
    
    dot.render('diagrams/generic_pipeline', cleanup=True)

# 4. Detector Comparison (YOLO vs. Faster R-CNN) - Simplified
def plot_detector_comparison():
    dot = create_graph('detector_comparison', 'Two-Stage vs. One-Stage Detectors', rankdir='LR')
    
    with dot.subgraph(name='cluster_one_stage') as c:
        c.attr(style='filled', color='#FDE2E4', label='One-Stage (e.g., YOLO, SSD)')
        c.node('Input1', 'Input Image')
        c.node('DirectPredict', 'Direct Prediction of\nBoxes & Classes', fillcolor='#B7E4C7')
        c.node('Output1', 'Final Detections')
        c.edge('Input1', 'DirectPredict')
        c.edge('DirectPredict', 'Output1')
        c.attr(rank='same') # Keep nodes on same level
    
    with dot.subgraph(name='cluster_two_stage') as c:
        c.attr(style='filled', color='#A9D6E5', label='Two-Stage (e.g., Faster R-CNN)')
        c.node('Input2', 'Input Image')
        c.node('RPN', 'Stage 1: Region Proposal Network\n(Suggests Regions)', fillcolor='#B7E4C7')
        c.node('ClassReg', 'Stage 2: Classifier + Regressor\n(Refines & Labels Regions)', fillcolor='#FDE2E4')
        c.node('Output2', 'Final Detections')
        c.edge('Input2', 'RPN')
        c.edge('RPN', 'ClassReg')
        c.edge('ClassReg', 'Output2')
        c.attr(rank='same') # Keep nodes on same level
        
    dot.node('Speed', 'Speed: Very Fast\nAccuracy: Good', shape='note', fillcolor='#F0FDF4')
    dot.node('Accuracy', 'Speed: Slower\nAccuracy: High', shape='note', fillcolor='#F0FDF4')
    
    dot.edge('Output1', 'Speed', style='invis') # Use invis to position without connecting
    dot.edge('Output2', 'Accuracy', style='invis')

    dot.render('diagrams/detector_comparison', cleanup=True)

# 5. YOLO Architecture Simplified for Graphviz
def plot_yolo_architecture_flow():
    dot = create_graph('yolo_architecture_flow', 'YOLO (You Only Look Once) Architecture Concept', rankdir='LR')
    
    dot.node('Input', 'Input Image', fillcolor='#A9D6E5')
    dot.node('Grid', 'Divide Image into Grid\n(e.g., 7x7)', fillcolor='#B7E4C7')
    dot.node('CNN', 'CNN Feature Extractor\n(for each Grid Cell)', fillcolor='#FDE2E4')
    dot.node('Predict', 'Each Cell Predicts:\n- BBox Coordinates (x,y,w,h)\n- Confidence Score\n- Class Probabilities', fillcolor='#B7E4C7')
    dot.node('NMS', 'NMS Post-processing', fillcolor='#FDE2E4')
    dot.node('Output', 'Final Detections', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Input', 'Grid')
    dot.edge('Grid', 'CNN')
    dot.edge('CNN', 'Predict')
    dot.edge('Predict', 'NMS')
    dot.edge('NMS', 'Output')
    
    dot.render('diagrams/yolo_architecture_flow', cleanup=True)

# 6. Training Workflow (for Object Detection)
def plot_training_workflow():
    dot = create_graph('training_workflow', 'Object Detection Training Workflow', rankdir='TB')
    
    dot.node('Start', 'Start Training', shape='oval', fillcolor='#A9D6E5')
    dot.node('Data', 'Load Batch of Images + Labels\n(Ground Truth Boxes/Classes)', shape='box')
    dot.node('Forward', 'Forward Pass\n(Model Predicts)', shape='box')
    dot.node('Loss', 'Calculate Loss\n(Difference between Prediction & GT)', shape='box', fillcolor='#FDE2E4')
    dot.node('Backward', 'Backward Pass\n(Compute Gradients)', shape='box')
    dot.node('Optimizer', 'Optimizer Step\n(Update Model Weights)', shape='box')
    dot.node('CheckStop', 'Stopping Criterion Met?\n(e.g., Epochs, Accuracy)', shape='diamond', fillcolor='#FDE2E4')
    dot.node('End', 'Model Trained', shape='oval', fillcolor='#2E86AB', fontcolor='white')
    
    dot.edge('Start', 'Data')
    dot.edge('Data', 'Forward')
    dot.edge('Forward', 'Loss')
    dot.edge('Loss', 'Backward')
    dot.edge('Backward', 'Optimizer')
    dot.edge('Optimizer', 'CheckStop')
    dot.edge('CheckStop', 'Data', label='No')
    dot.edge('CheckStop', 'End', label='Yes')
    
    dot.render('diagrams/training_workflow', cleanup=True)

if __name__ == "__main__":
    plot_detection_definition()
    plot_nms_flow()
    plot_generic_pipeline()
    plot_detector_comparison()
    plot_yolo_architecture_flow()
    plot_training_workflow()
