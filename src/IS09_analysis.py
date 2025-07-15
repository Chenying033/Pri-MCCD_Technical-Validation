# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from glob import glob
# from scipy.stats import f_oneway
# from matplotlib import rcParams
# from matplotlib import font_manager
#
# 
# TNR_PATH = '/data1/cy/TimesNewRoman.ttf'
# times_new_roman_font = font_manager.FontProperties(fname=TNR_PATH)
#
# #
# rcParams['font.family'] = times_new_roman_font.get_name()
# rcParams['axes.unicode_minus'] = False  
#
#

#
# 
# INPUT_DIR = '/data1/Multimodal-Infomax/IS09NPZ/test'
#

# OUTPUT_DIR = '/data1/Multimodal-Infomax/climate_validation_external_labels'
#
# (0-based)
# FEATURE_MAPPING = {
#     'pcm_RMSenergy_sma_amean': {'index': 6, 'name': 'Mean Energy'},
#     'pcm_RMSenergy_sma_stddev': {'index': 10, 'name': 'Energy StdDev'},
#     'F0_sma_stddev': {'index': 186, 'name': 'Pitch Variation (F0 StdDev)'},
#     'voiceProb_sma_amean': {'index': 174, 'name': 'Voicing Probability'},
#     'pcm_zcr_sma_amean': {'index': 162, 'name': 'Mean Zero-Crossing Rate'},
#     'F0_sma_amean': {'index': 182, 'name': 'Mean Pitch (F0 Mean)'}
# }
#
# # label
# LABEL_MAP = {
#     '0': 'Negative',
#     '1': 'Neutral',
#     '2': 'Positive'
# }
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# print(f"结果将保存在: {OUTPUT_DIR}")

# all_features_list = []
# all_labels = []
# npy_files = glob(os.path.join(INPUT_DIR, '*.npy'))
#
# if not npy_files:
#     print(f"错误：在文件夹 '{INPUT_DIR}' 中没有找到任何 .npy 文件。请检查路径。")
#     exit()
#
# print(f"找到 {len(npy_files)} 个 .npy 文件，开始加载...")
#
# for file_path in npy_files:
#     try:
#         # 从文件名解析标签
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         label_key = base_name.split('_')[-1]
#
#         if label_key in LABEL_MAP:
#             label = LABEL_MAP[label_key]
#
#             # 加载特征
#             features = np.load(file_path)
#             if features.shape[-1] == 384:
#                 all_features_list.append(features.flatten())
#                 all_labels.append(label)
#             else:
#                 print(f"警告：文件 {os.path.basename(file_path)} 的形状不符合预期 {features.shape}，已跳过。")
#         else:
#             print(f"警告：无法从文件名 {os.path.basename(file_path)} 解析标签，已跳过。")
#
#     except Exception as e:
#         print(f"警告：处理文件 {os.path.basename(file_path)} 时出错: {e}，已跳过。")
#
# if not all_features_list:
#     print("错误：未能成功加载任何有效的特征数据和标签。")
#     exit()
#

# all_features_matrix = np.vstack(all_features_list)
# print(f"数据加载完成。总样本数: {all_features_matrix.shape[0]}, 特征维度: {all_features_matrix.shape[1]}")
#
# # ---  Pandas DataFrame--
# data_for_df = {}
# for key, info in FEATURE_MAPPING.items():
#     data_for_df[info['name']] = all_features_matrix[:, info['index']]
#
# df = pd.DataFrame(data_for_df)
# df['Climate'] = all_labels  
#

# climate_counts = df['Climate'].value_counts(normalize=True).sort_index()
# print("\n课堂氛围分组数量占比 (基于文件名):")
# print(climate_counts)
#
# # --- -
# print("\n开始生成图表和统计报告...")
# report_lines = ["Feature Validation Report\n", "=" * 30, "\n"]
#

# sns.set_theme(style="whitegrid")
#
# for key, info in FEATURE_MAPPING.items():
#     feature_name_for_plot = info['name']
#
#     plt.figure(figsize=(13, 7))
#

#     ax = sns.boxplot(
#         x='Climate',
#         y=feature_name_for_plot,
#         data=df,
#         order=['Negative', 'Neutral', 'Positive'],
#         palette="viridis"
#     )
#

#     groups = [df[feature_name_for_plot][df['Climate'] == g] for g in ['Negative', 'Neutral', 'Positive']]
#     
#     valid_groups = [g for g in groups if not g.empty]
#
#     p_value_text = "N/A"
#     if len(valid_groups) > 1:
#         f_stat, p_value = f_oneway(*valid_groups)
#         p_value_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
#         report_lines.append(f"- {feature_name_for_plot}: ANOVA F-statistic={f_stat:.2f}, {p_value_text}\n")
#

#     plt.title(f'Distribution of {feature_name_for_plot} across Climate Groups\n(ANOVA: {p_value_text})', fontsize=28)
#     plt.xlabel('Ground Truth Climate Group', fontsize=24)
#     plt.ylabel(f'Value of {feature_name_for_plot}', fontsize=24)
#
#     ax.ticklabel_format(style='plain', axis='y')
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#
#     
#     output_path = os.path.join(OUTPUT_DIR, f'boxplot_{key}.png')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#
#     print(f"  - 图表已保存: {output_path}")
#

# report_path = os.path.join(OUTPUT_DIR, 'statistical_report.txt')
# with open(report_path, 'w') as f:
#     f.writelines(report_lines)
# print(f"  - 统计报告已保存: {report_path}")
#
# print("\n所有分析和可视化已完成！")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from matplotlib import rcParams
from matplotlib import font_manager


TNR_PATH = '/data1/cy/TimesNewRoman.ttf'
font_prop = font_manager.FontProperties(fname=TNR_PATH)
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False  



INPUT_DIR = '/data1/Multimodal-Infomax/IS09NPZ/test'
OUTPUT_DIR = '/data1/Multimodal-Infomax/climate_composition_plots'  
FEATURE_MAPPING = {
    'pcm_RMSenergy_sma_amean': {'index': 6, 'name': 'Mean Energy'},
    'pcm_RMSenergy_sma_stddev': {'index': 10, 'name': 'Energy StdDev'},
    'F0_sma_stddev': {'index': 186, 'name': 'Pitch Variation (F0 StdDev)'},
    'voiceProb_sma_amean': {'index': 174, 'name': 'Voicing Probability'},
    'pcm_zcr_sma_amean': {'index': 162, 'name': 'Mean Zero-Crossing Rate'},
    'F0_sma_amean': {'index': 182, 'name': 'Mean Pitch (F0 Mean)'}
}
LABEL_MAP = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}
NUM_BINS = 5  

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"图表将保存在: {OUTPUT_DIR}")


all_features_list = []
all_labels = []
npy_files = glob(os.path.join(INPUT_DIR, '*.npy'))
if not npy_files:
    print(f"错误：在文件夹 '{INPUT_DIR}' 中没有找到任何 .npy 文件。")
    exit()

for file_path in npy_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    label_key = base_name.split('_')[-1]
    if label_key in LABEL_MAP:
        label = LABEL_MAP[label_key]
        features = np.load(file_path)
        if features.shape[-1] == 384:
            all_features_list.append(features.flatten())
            all_labels.append(label)

all_features_matrix = np.vstack(all_features_list)
data_for_df = {info['name']: all_features_matrix[:, info['index']] for key, info in FEATURE_MAPPING.items()}
df = pd.DataFrame(data_for_df)
df['Climate'] = all_labels

print(f"数据加载完成，总样本数: {len(df)}")


print("\n开始生成堆叠柱状图...")


color_palette = {'Negative': 'skyblue', 'Neutral': 'orange', 'Positive': 'salmon'}
climate_order = ['Negative', 'Neutral', 'Positive']

for key, info in FEATURE_MAPPING.items():
    feature_name = info['name']

    
    try:
        df[f'{feature_name}_bin'] = pd.qcut(df[feature_name], q=NUM_BINS, duplicates='drop', labels=False)

        bin_labels = pd.qcut(df[feature_name], q=NUM_BINS, duplicates='drop').cat.categories
    except ValueError:
       
        df[f'{feature_name}_bin'] = pd.cut(df[feature_name], bins=NUM_BINS, labels=False)
        bin_labels = pd.cut(df[feature_name], bins=NUM_BINS).cat.categories


    cross_tab = pd.crosstab(df[f'{feature_name}_bin'], df['Climate'])


    cross_tab = cross_tab.reindex(columns=climate_order, fill_value=0)


    cross_tab_percent = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100


    fig, ax = plt.subplots(figsize=(14, 7))

    cross_tab_percent.plot(
        kind='bar',
        stacked=True,
        color=[color_palette[col] for col in cross_tab_percent.columns],
        ax=ax,
        width=0.7  
    )


    plt.title(f'Climate Composition within {feature_name} Ranges', fontsize=28, pad=20)
    plt.xlabel(f'Range of {feature_name}', fontsize=24)
    plt.ylabel('Percentage of Samples (%)', fontsize=24)
    plt.xticks(rotation=10, ha='center', fontsize=22)  
    plt.yticks(fontsize=22)
    plt.ylim(0, 115)  

  
    ax.set_xticklabels([f"{interval.left:.2f} - {interval.right:.2f}" for interval in bin_labels], rotation=10)

  
    for c in ax.containers:
        labels = [f'{w:.1f}%' if w > 5 else '' for w in c.datavalues]  
        ax.bar_label(c, labels=labels, label_type='center', fontsize=18, color='black', weight='bold')

 
    bin_counts = cross_tab.sum(axis=1)
    for i, count in enumerate(bin_counts):
        ax.text(i, 102, f'n={count}', ha='center', va='bottom', fontsize=11, color='dimgray')


    ax.legend(
        title='Climate',
        title_fontsize=18,  
        fontsize=16,  
        bbox_to_anchor=(1.02, 1.0),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 


    output_path = os.path.join(OUTPUT_DIR, f'stacked_bar_{key}.png')
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"  - 堆叠柱状图已保存: {output_path}")

print("\n所有堆叠柱状图已生成！")