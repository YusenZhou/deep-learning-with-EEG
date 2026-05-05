import sys
from src.train import train_on_A1,train_all_subjects
from src.plot import plot_results_A1,plot_A1_confusion_matrix,plot_bar_chart
from src.ablation import ablation_heads,ablation_kern_len,ablation_layers

functions = {
    'train_on_A1': train_on_A1,
    'train_all_subjects': train_all_subjects,
    'plot_results_A1': plot_results_A1,
    'plot_A1_confusion_matrix': plot_A1_confusion_matrix,
    'plot_bar_chart': plot_bar_chart,
    'ablation_heads': ablation_heads,
    'ablation_kern_len': ablation_kern_len,
    'ablation_layers': ablation_layers,
}

if len(sys.argv) == 1:
    print('available commands')
    for func in functions:
        print(f'  python run.py {func}')
elif len(sys.argv) == 2:
    if sys.argv[1] in functions:
        functions[sys.argv[1]]()
    else:
        print('wrong function')
else:
    print('wrong command')