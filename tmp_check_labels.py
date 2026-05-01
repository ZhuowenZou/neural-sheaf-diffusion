from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from unittest.mock import patch

with patch('builtins.input', return_value='y'):
    dataset = PyGNodePropPredDataset(name='tgbn-trade', root='datasets')

dataset.reset_label_time()
td = dataset.get_TemporalData()
seen = 0
for ts in sorted(set(td.t.tolist())):
    out = dataset.get_node_label(ts)
    if out is not None:
        label_ts, label_srcs, labels = out
        print('first non-empty ts', ts)
        print('label_ts shape', label_ts.shape)
        print('label_srcs shape', label_srcs.shape)
        print('labels shape', labels.shape)
        print('src sample', label_srcs[:10].tolist())
        print('labels sample', labels[:10].tolist())
        break
    seen += 1
    if seen > 200:
        print('no labels in first 200 timestamps')
        break
