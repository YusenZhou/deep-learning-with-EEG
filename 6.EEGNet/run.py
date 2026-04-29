import sys
from src.train_models import train_on_A1,train_on_all_data
from src.plot import plot_A1_result,plot_A1_confusion_matrix,plot_bar_chart,show_summary,visualize_filters
from src.ablation import ablation_depth_multiplier,ablation_dropout,ablation_kernel_length

functions = {
    'train_on_A1':train_on_A1,
    'train_on_all_data':train_on_all_data,
    'plot_A1_result':plot_A1_result,
    'plot_A1_confusion_matrix':plot_A1_confusion_matrix,
    'plot_bar_chart':plot_bar_chart,
    'show_summary':show_summary,
    'visualize_filters':visualize_filters,
    'ablation_depth_multiplier':ablation_depth_multiplier,
    'ablation_dropout':ablation_dropout,
    'ablation_kernel_length':ablation_kernel_length
}
if len(sys.argv) == 1:
    print('available commands')
    for func in functions:
        print(f'python run.py {func}')
elif len(sys.argv) == 2:
    if sys.argv[1] in functions:
        functions[sys.argv[1]]()
    else:
        print('wrong function')
else:
    print('wrong command')