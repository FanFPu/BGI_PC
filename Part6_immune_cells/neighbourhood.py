import os, sys
# del sys.path[4]
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
import squidpy as sq
from SPACEL.setting import set_environ_seed
set_environ_seed(42)

args = sys.argv
number = int(args[1])

od = '/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_{}_Immune_EPi'.format(number)
os.system(f'mkdir -p {od}')
os.chdir(od)

adata= sc.read("/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/Splane/n_cluster_15_n_neighbors_8_k_2_gnn_dropout_0.5/Spoint_anno.h5ad")
adata = adata[adata.obs['batch']=='HPCP2']



sc.set_figure_params(facecolor="white", figsize=(4, 4))
sc.settings.verbosity = 3
sc.settings.dpi = 300
sc.settings.figdir = "./figures"



# adata = adata[adata.obs['annotated_cluster'].isin(['Tum_type1','Tum_type2','Tum_type3','Tum_type4','Tum_type5','Tum_type6'])]
# cluster_annotation = {'11':'SD11','2':'SD2','10':'SD10','0':'SD0','3':'SD3','14':'SD14'}
# # adata.obs['region'] = adata.obs['spatial_domain'].map(cluster_annotation).astype('category')
# adata = adata[adata.obs['region'].isin(['SD11','SD2','SD10','SD0','SD3','SD14'])]


adata.obs['annotated_cluster1'] = adata.obs['annotated_cluster1'].astype('str')
# 创建一个包含需要处理的Tum_type列表
# tumor_types = ["Tum_type1", "Tum_type2", "Tum_type3", "Tum_type4", "Tum_type5", "Tum_type6"]
tumor_types = [ "Tum_type2", "Tum_type3", "Tum_type4"]
for tumor_type in tumor_types:
    adata.obs.loc[adata.obs['annotated_cluster1'] == tumor_type, 'region'] = tumor_type
# 检查结果
other_types = ['cDC2_CLEC10A',  'cDC1_CLEC9A',  'Classical_PlasmaB',  'Naive_B_IGHD', 'PlasmaB_SDC1', 'MemoryB_MS4A1', 'cDC3_LAMP3','Macro_LILRB1', 'Macro_APOE','Macro_LYVE1',  'Naive T cell', 'Treg', 'CTL']
# 使用循环为每个Tum_type设置对应的Tumsubtype
for tumor_type in other_types:
    adata.obs.loc[adata.obs['annotated_cluster1'] == tumor_type, 'fibsubtype'] = tumor_type


print(adata.obs['region'].value_counts())
adata.obs['region'] = adata.obs['region'].astype('category')
adata.obs['fibsubtype'] = adata.obs['fibsubtype'].astype('category')

existing_categories = adata.obs['region'].cat.categories
adata.obs['region'].cat.add_categories(['other'], inplace=True)
adata.obs['region'] = adata.obs['region'].fillna('other')
existing_categories = adata.obs['fibsubtype'].cat.categories
adata.obs['fibsubtype'].cat.add_categories(['other'], inplace=True)
adata.obs['fibsubtype'] = adata.obs['fibsubtype'].fillna('other')


# 计算邻居
# sc.pp.neighbors(adata, n_neighbors=30, use_rep='spatial')
# # 计算空间邻近的细胞
sq.gr.spatial_neighbors(adata, coord_type='generic', library_key='batch', radius=number)
adata.write("Immune_EPi_neighbors_HPCP2_new.h5ad")



# 计算每个细胞周围不同fibsubtype的邻居占比
def calculate_fibsubtype_proportion(adata):
    # 获取connectivities
    connectivities = adata.obsp['spatial_connectivities'].tocsr()  # 确保是csr_matrix格式
    # 初始化一个空的DataFrame来存储结果
    result = pd.DataFrame(index=adata.obs['region'].unique(), columns=adata.obs['fibsubtype'].unique())
    result[:] = np.nan
    
    # 获取所有细胞的索引
    cell_indices = np.arange(len(adata))
    
    for region in adata.obs['region'].unique():
        for fibsubtype in adata.obs['fibsubtype'].unique():
            # 找到当前region和fibsubtype的细胞索引
            cells_in_region = cell_indices[(adata.obs['region'] == region)]
            different_fibsubtype_count = 0
            total_neighbors = 0
            
            for cell in cells_in_region:
                # 获取当前细胞的邻居索引
                start = connectivities.indptr[cell]
                end = connectivities.indptr[cell + 1]
                neighbors_of_cell = connectivities.indices[start:end]
                neighbors_fibsubtype = adata.obs['fibsubtype'].iloc[neighbors_of_cell]
                # 计算不同fibsubtype的邻居数量
                different_fibsubtype_count += (neighbors_fibsubtype == fibsubtype).sum()
                total_neighbors += len(neighbors_of_cell)
                
            if total_neighbors > 0:
                result.at[region, fibsubtype] = different_fibsubtype_count / total_neighbors
    return result

result_df = calculate_fibsubtype_proportion(adata)
print(result_df)
# # # 计算空间邻近的细胞

result_df_normalized = result_df.drop('other', axis=1)
result_df_normalized = result_df_normalized.div(result_df_normalized.abs().sum(axis=0), axis=1)

for col in result_df_normalized.columns:
    result_df_normalized[col] = pd.to_numeric(result_df_normalized[col], errors='coerce')
result_df_normalized = result_df_normalized.T
order = ['Tum_type2', 'Tum_type3', 'Tum_type4']
result_df_normalized = result_df_normalized[order]

import seaborn as sns
sns.set_style('whitegrid')
fig,ax = plt.subplots(figsize = (3.5,6))
#sns.heatmap(df, cmap='YlGn',center = 0,ax=ax,linewidths=0.3,vmin=0,vmax=1)  # 绘制有色数据时将色彩映射居中的值
sns.heatmap(result_df_normalized, cmap='coolwarm',center = 0,ax=ax,linewidths=0.3,vmin=0,vmax=result_df_normalized.max().max())  # 绘制有色数据时将色彩映射居中的值
ax.set_facecolor('#dcdcdc')
plt.savefig('nhood_Epi_Immune_HPCP2_new.pdf', dpi=300, bbox_inches='tight')






# existing_categories = ['APOE_Macrophages', 'B_cells', 'CA10_T_cells', 'CD1C_cDC2', 'CD8A_CTL',
#        'CLEC9A_cDC1', 'LAMP3_cDC3', 'LYVE1_Macrophages', 'Mast_cells',
#        'NCAM1_NK_cells', 'NLRP3_Macrophages', 'Plasma_cells', 'TCF7_T_cells',
#        'TNFRSF9_T_cells', 'TOP2A_T_cells']

adata = sc.read('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_30um/Immune_EPi_neighbors_HPCP2.h5ad')
existing_categories = ['Tum_type2', 'Tum_type3', 'Tum_type4']

connectivities = adata.obsp['spatial_connectivities'].tocsr()
# 获取所有细胞的索引
cell_indices = np.arange(len(adata))
st_ad_list = []

for immune in existing_categories:
    cells_in_region = cell_indices[(adata.obs['annotated_cluster1'] == immune)]
    indices = []
    for cell in cells_in_region:
        # 获取当前细胞的邻居索引
        start = connectivities.indptr[cell]
        end = connectivities.indptr[cell + 1]
        neighbors_of_cell = connectivities.indices[start:end]
        # 检查邻居细胞的'annotated_cluster'是否在指定的类别中
        valid_neighbors = neighbors_of_cell[adata.obs['annotated_cluster1'].iloc[neighbors_of_cell].isin(['cDC2_CLEC10A',  'cDC1_CLEC9A',  'Classical_PlasmaB',  'Naive_B_IGHD', 'PlasmaB_SDC1', 'MemoryB_MS4A1', 'cDC3_LAMP3','Macro_LILRB1', 'Macro_APOE','Macro_LYVE1',  'Naive T cell', 'Treg', 'CTL'])]
        indices.extend(valid_neighbors.tolist())
    indices = list(set(indices))
    adata_select = adata[indices]
    adata_select.obs['tumor_type'] = immune
    st_ad_list.append(adata_select)

adata = sc.AnnData.concatenate(*st_ad_list, join='outer')
adata.write('HPCP2_take_type2_nei_Imm_nei.h5ad')

import pandas as pd
# 假设 adata 是你的 AnnData 对象
df = adata.obs
# 计算 tumor_type 和 annotated_cluster 的联合分布
joint_distribution = pd.crosstab(df['tumor_type'], df['annotated_cluster1'])
joint_distribution_transposed = joint_distribution.T
output_path = "joint_distribution.csv"  # 修改为你希望的保存路径
print(joint_distribution_transposed)
joint_distribution_transposed.to_csv(output_path)















import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


H100 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_100_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H100 = H100.reset_index()

H200 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_200_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H200 = H200.reset_index()

H400 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_400_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H400 = H400.reset_index()

H600 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_600_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H600 = H600.reset_index()

H800 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_800_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H800 = H800.reset_index()

H60 = pd.read_csv('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/neigh_60_Immune_EPi/joint_distribution.csv', index_col=0)  # 使用 index_col=0 将第一列作为索引
H60 = H60.reset_index()
# 提取 `annotated_cluster1` 和 `Tum_type2` 列并合并
data = pd.DataFrame({
    "annotated_cluster1": H100["annotated_cluster1"],
    "30um": H60["Tum_type2"],
    "50um": H100["Tum_type2"],
    "100um": H200["Tum_type2"],
    "200um": H400["Tum_type2"],
    "300um": H600["Tum_type2"],
    "400um": H800["Tum_type2"]
})



# def calculate_tumor_type_proximity(adata, tumor_types, cell_types):
#     # 获取connectivities
#     connectivities = adata.obsp['spatial_connectivities'].tocsr()  # 确保是csr_matrix格式
#     result = pd.DataFrame(index=tumor_types, columns=cell_types)
#     result[:] = np.nan

#     cell_indices = np.arange(len(adata))
    
#     # 遍历每个Tumor类型
#     for tumor_type in tumor_types:
#         tumor_cells = cell_indices[adata.obs['annotated_cluster'] == tumor_type]
        
#         for cell_type in cell_types:
#             proximity_count = 0
#             total_neighbors = 0
            
#             for cell in tumor_cells:
#                 # 获取当前细胞的邻居索引
#                 start = connectivities.indptr[cell]
#                 end = connectivities.indptr[cell + 1]
#                 neighbors_of_cell = connectivities.indices[start:end]
#                 neighbors_cell_type = adata.obs['annotated_cluster'].iloc[neighbors_of_cell]
                
#                 # 计算邻近的cell_type的邻居数量
#                 proximity_count += (neighbors_cell_type == cell_type).sum()
#                 total_neighbors += len(neighbors_of_cell)
            
#             # 计算邻近比例
#             if total_neighbors > 0:
#                 result.at[tumor_type, cell_type] = proximity_count / total_neighbors
    
#     return result

# # 选择cell_types，考虑其他非肿瘤类型
# cell_types = ['Fibroblast', 'Sensory_Neuron', 'Endothelial', 'Plasmocyte', 'Macrophage', 'Tcell', 'Bcell', 'Melanocytes']

# # 计算各个Tum_type与其他细胞类型的邻近比例
# tumor_types = ['Tum_type1', 'Tum_type2', 'Tum_type3', 'Tum_type4', 'Tum_type5', 'Tum_type6']
# proximity_result = calculate_tumor_type_proximity(adata, tumor_types, cell_types)








# 确保所有数据都是数值类型
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
# 设置横坐标，时间点数据作为列名
time_points = ['30um', '50um', '100um', '200um', '300um','400um']
# 设置细胞类型（纵坐标）
cell_types = data['annotated_cluster1']
# 设置为13个颜色，每个细胞类型一个颜色
colors = ["#FF0000","#00FF00","#ffff1a","#B87A3D",
"#E78AC3","#679966","#6b42c8","#43D9FE",
"#1a1aff","#E5C494","#333399","#CCE5FF","#d0cdb8"]



#===========细胞数量图
# 绘制堆叠柱状图
plt.figure(figsize=(7, 6))
bottoms = np.zeros(len(time_points))  # 初始化底部位置，确保是 float 类型
bar_width = 0.4  # 每个柱子的宽度更宽

# 遍历每个细胞类型
for i, cell_type in enumerate(cell_types):
    # 提取该细胞类型的数据
    cell_data = data.iloc[i, 1:].values.astype(float)  # 确保 cell_data 是 float 类型
    # 绘制每个细胞类型在不同时间点的数量，堆叠起来
    plt.bar(time_points, cell_data, label=cell_type, 
            bottom=bottoms, width=bar_width, color=colors[i % len(colors)])  # 增大柱宽
    # 更新底部位置
    bottoms += cell_data  # 更新底部位置以便下一个细胞类型堆叠
# 添加加粗的标签和标题
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Cell Count", fontsize=12, fontweight="bold")
plt.title("Immune Subtype Distribution Across Tumor Types", fontsize=14, fontweight="bold")
# 加粗图例的标题
plt.legend(title="Cell Types", title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# 调整布局
plt.tight_layout()
# 保存图形为文件
plt.savefig("scell_type_distribution_bold.png", dpi=300)  # 保存为 PNG 格式，分辨率为300 dpi
# 显示图形
plt.show()



#比例图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 转置数据用于计算每列的比例
data_transposed = data.set_index("annotated_cluster1").T
data_transposed_normalized = data_transposed.div(data_transposed.sum(axis=1), axis=0)  # 每列归一化
# 准备绘图数据
time_points = data_transposed_normalized.index  # 横轴是时间点
cell_types = data["annotated_cluster1"]         # 细胞类型是堆叠部分
# colors = plt.cm.tab20.colors[:len(cell_types)]  # 为每种细胞类型分配颜色
# 绘制堆叠柱状图
plt.figure(figsize=(10, 8))
bottoms = np.zeros(len(time_points))  # 初始化底部位置
for i, cell_type in enumerate(cell_types):
    # 提取当前细胞类型在所有列的比例
    cell_data = data_transposed_normalized[cell_type].values
    # 绘制堆叠柱
    plt.bar(time_points, cell_data, label=cell_type, 
            bottom=bottoms, color=colors[i % len(colors)], width=0.8)
    bottoms += cell_data  # 更新底部位置
# # 添加前三占比最大的标注
# for idx, time_point in enumerate(time_points):
#     # 提取每个时间点的比例，并排序
#     proportions = data_transposed_normalized.iloc[idx]
#     sorted_proportions = proportions.sort_values(ascending=False)
#     top3 = sorted_proportions[:3]  # 取前三
#     # 在图上标注前三的比例
#     for j, (cell_type, proportion) in enumerate(top3.items()):
#         plt.text(
#             x=time_point, 
#             y=sum(sorted_proportions[:j]),  # 累加高度确定文本位置
#             s=f"{proportion:.2f}", 
#             fontsize=10, ha='center', va='center', color="darkblue", fontweight="bold"
#         )
# 添加标签和标题，字体设置
plt.xlabel("Distance", fontsize=14, fontweight="bold")  # 横轴标签
plt.ylabel("Proportion", fontsize=14, fontweight="bold")  # 纵轴标签
plt.title("Proportion of Cell Types Across Distances", fontsize=16, fontweight="bold")  # 图标题
# 设置x轴和y轴刻度字体加粗
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
# 绘制图例并调整字体样式
legend = plt.legend(
    title="Cell Types", 
    title_fontsize=14,  # 调整标题字体大小
    fontsize=12,  # 图例内容字体大小
    prop={'weight': 'bold'},  # 图例内容加粗
    loc='upper left', 
    bbox_to_anchor=(1.05, 1)  # 图例移到图外右上角
)
legend.get_title().set_fontweight("bold")
# 调整布局
plt.grid(False)
plt.tight_layout()
plt.savefig("/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2/cell_type_proportions1_updated.pdf", dpi=300)
# 显示图形
plt.show()







os.chdir('/hsfscqjf1/ST_CQ/P23Z28400N0255/fanfengpu/PC/Paper/nhood_diffum/HPCP2')
#单个细胞类型的比例图
data_transposed = data.set_index("annotated_cluster1").T
data_transposed_normalized = data_transposed.div(data_transposed.sum(axis=1), axis=0)  # 每列归一化
# 提取 Macro_APOE 数据
macro_apoe_proportions = data_transposed_normalized["Macro_APOE"]
# classical_plasmab_proportions = data_transposed_normalized["Classical_PlasmaB"]
classical_plasmab_proportions = data_transposed_normalized["PlasmaB_SDC1"]
 
bar_width = 0.4
offset = bar_width / 2
plt.figure(figsize=(8, 6))
# 绘制 Macro_APOE 的柱子
x_positions_apoe = np.arange(len(macro_apoe_proportions.index)) - offset
plt.bar(
    x_positions_apoe, macro_apoe_proportions.values, 
    label="Macro_APOE", color='#ffa62b', width=bar_width
)
# 绘制 Classical_PlasmaB 的柱子
x_positions_plasmab = np.arange(len(classical_plasmab_proportions.index)) + offset
plt.bar(
    x_positions_plasmab, classical_plasmab_proportions.values, 
    label="PlasmaB_SDC1", color='#87a922', width=bar_width
)
# 设置刻度和标签
plt.xticks(np.arange(len(macro_apoe_proportions.index)), macro_apoe_proportions.index, fontsize=11)
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Proportion", fontsize=12, fontweight="bold")
plt.title("Proportion of Selected Cell Types Across Distances", fontsize=14, fontweight="bold")
# 单独添加图例在右上方
plt.legend(
    title="Cell Types", fontsize=10, title_fontsize=12, 
    loc='upper left', bbox_to_anchor=(1, 1)
)
# 调整布局
plt.grid(False)
plt.tight_layout()
# 保存图形
plt.savefig("selected_cell_types_proportions1.pdf", dpi=300)
# 显示图形
plt.show()
