import networkx as nx

def build_graph():
    G = nx.DiGraph()
    emotions = {
        'sadness': ['journaling', 'talk to someone'],
        'anxiety': ['deep breathing', 'mindfulness'],
        'anger': ['walk', 'deep breathing']
    }
    for emotion, strategies in emotions.items():
        G.add_node(emotion, type='emotion')
        for strategy in strategies:
            G.add_node(strategy, type='strategy')
            G.add_edge(emotion, strategy, relation='helps_with')
    return G

def get_strategies_from_graph(emotion, graph):
    if emotion not in graph:
        return []
    return [tgt for src, tgt in graph.edges() if src == emotion]