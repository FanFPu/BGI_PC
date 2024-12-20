#%%
import os,sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import anndata as an
import matplotlib as mpl
import seaborn as sns
import scanpy.external as sce
import glob
Image.MAX_IMAGE_PIXELS = None
from SPACEL.setting import set_environ_seed
set_environ_seed(42)
from SPACEL import Spoint
from scipy.sparse import csr_matrix
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.serif'] = ['Arial']
sc.settings.set_figure_params(dpi=50,dpi_save=300,facecolor='white',fontsize=10,vector_friendly=True,figsize=(3,3))
sc.settings.verbosity = 3
#%%

### Processing of scRNA-Seq data
sc_ad = sc.read('sc_all.h5ad')
sc_ad.X = sc_ad.layers['counts']
sc.pp.filter_genes(sc_ad,min_cells=1)
sc.pp.filter_cells(sc_ad,min_genes=1)
sc_ad.obs['celltype'] = sc_ad.obs['celltype'].astype('category')
sc.pp.log1p(sc_ad)

def Spoint_cellbin(st_ad,sc_ad):
    # %% [markdown]
    # In this step, we initialize Spoint model with anndata objects for scRNA-seq and ST as input. The parameter `celltype_key` is the column name of cell type annotation in `.obs`. The parameter `sm_size` controls the number of simulated spots.

    # %%
    spoint = Spoint.init_model(sc_ad,st_ad,celltype_key='celltype', sm_size=500000, n_threads=30)

    # %% [markdown]
    # Here, we train the model to obtain the best model for cell type deconvolution. The parameter "max_steps" is the maximum number of steps in the training process. Generally, the model will stop training before reaching the maximum number of steps.

    # %%
    spoint.train(max_steps=5000, batch_size=4096)

    # %% [markdown]
    # Then, we use the trained model to predict the cell type composition of each spot in ST. It will return a `DataFrame` object, where each row represents a spot in ST, each column represents a cell type in scRNA-seq, and each item represents the proportion of a cell type in a spot. Also, we can get the anndata of the ST of the data frame with the deconvolution result embedded in `.obs`.

    # %%
    pre = spoint.deconv_spatial()

    # %%
    st_ad = spoint.st_ad
    #st_ad.__dict__['_raw'].__dict__['_var'] = st_ad.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})


    # %%
    #st_ad.write(f'Spiont_{sample}.h5ad')
    st_ad.write('Spoint.h5ad')

    # %%
    sc.pl.spatial(st_ad,color=pre.columns,ncols=6,vmax='p99',spot_size=30,save='embeddings.pdf')


### Run the function using Spatial transcriptome data 
st_ad = sc.read('raw.h5ad')
od = 'raw/'
os.system(f'mkdir -p {od}')
os.chdir(od)
Spoint_cellbin(st_ad,sc_ad)

def getDefaultColors(n, type = 1):
    if type == 1:
        colors = [
            "#ff8d1a", "#7cd5c8", "#c49a3f", "#5d8d9c", "#90353b",
            "#ff1a1a", "#1B9E77", "#1a1aff", "#ffff1a", "#ff1aff",
                "#507d41", "#502e71", "#1aff1a", "#c5383c", "#0081d1",
                "#674c2a", "#c8b693", "#aed688", "#f6a97a", "#c6a5cc",
                "#798234", "#6b42c8", "#cf4c8b", "#666666", "#ffd900",
                "#feb308", "#cb7c77", "#68d359", "#6a7dc9", "#c9d73d"]
    elif type == 2:
        if n <= 14:
            colors = ["#CCE5FF", "#B87A3D", "#8A2BE2", "#FEC643", "#FF99FF",
                      "#679966", "#663300", "#C71585", "#437BFE", "#FF0000", "#CCCC00",
                       "#43D9FE", "#00FF00", "#333399", "#43FE69", "#E5C494"]
        elif n <= 20:
            colors = ["#FFFFFF", "#d5492f", "#6bd155", "#683ec2", "#c9d754",
                  "#d04dc7", "#81d8ae", "#d34a76", "#607d3a", "#6d76cb",
                  "#ce9d3f", "#81357a", "#d3c3a4", "#3c2f5a", "#b96f49",
                  "#4e857e", "#6e282c", "#d293c8", "#393a2a", "#997579"]
        elif n <= 30:
            colors = ["#1aff1a", "#a03259", "#4836be", "#ffff1a", "#ff1aff",
                  "#ff8d1a", "#7cd5c8", "#c49a3f", "#da4f1e", "#90353b",
                  "#507d41", "#502e71", "#1B9E77", "#c5383c", "#0081d1",
                  "#674c2a", "#c8b693", "#aed688", "#f6a97a", "#c6a5cc",
                  "#FFFFFF", "#6b42c8", "#cf4c8b", "#666666", "#1a1aff",
                  "#feb308", "#cb7c77", "#68d359", "#8f5b5c", "#c9d73d"]
        else:
            colors = ["#1aff1a", "#a03259", "#4836be", "#ffff1a", "#ff1aff",
                  "#ff8d1a", "#7cd5c8", "#c49a3f", "#da4f1e", "#90353b",
                  "#507d41", "#502e71", "#1B9E77", "#c5383c", "#0081d1",
                  "#674c2a", "#c8b693", "#aed688", "#f6a97a", "#c6a5cc",
                  "#798234", "#6b42c8", "#cf4c8b", "#666666", "#1a1aff",
                  "#feb308", "#cb7c77", "#68d359", "#8f5b5c", "#c9d73d",
                  "#982f29", "#5ddb53", "#8b35d6", "#a9e047", "#ffd900",
                  "#e0dc33", "#d248d5", "#61a338", "#9765e5", "#69df96",
                  "#7f3095", "#d0d56a", "#371c6b", "#cfa738", "#5066d1",
                  "#e08930", "#83e6d6", "#df4341", "#6a8bd3", "#5d8d9c",
                  "#6ebad4", "#e34c75", "#50975f", "#d548a4", "#badb97",
                  "#b377cf", "#899140", "#564d8b", "#ddb67f", "#292344",
                  "#d0cdb8", "#421b28", "#5eae99", "#ff1a1a", "#406024",
                  "#e598d7", "#343b20", "#bbb5d9", "#975223", "#576e8b",
                  "#d97f5e", "#253e44", "#de959b", "#417265", "#712b5b",
                  "#8c6d30", "#a56c95", "#5f3121", "#8f846e", "#6a7dc9"]
    elif type == 3:
        colors = ["#588dd5", "#c05050", "#07a2a4", "#f5994e",
                "#9a7fd1", "#59678c", "#c9ab00", "#7eb00a"]
    elif type == 4:
        colors = ["#FC8D62", "#66C2A5", "#8DA0CB", "#E78AC3",
                "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]    
    elif type == 5:
        colors = ["#c14089", "#6f5553", "#E5C494", "#738f4c",
                "#bb6240", "#66C2A5", "#2dfd29", "#0c0fdc"]
    if n:
        if n <= len(colors):
            colors = colors[0:n]
        else:
            step = 16777200 // (n - len(colors)) - 2
            add_colors = []
            tmp = random.sample(range(step),1)[0]
            for i in range(n-len(colors)):
                hextmp = str(hex(tmp)).replace("0x",'')
                if len(hextmp) == 5:
                    hextmp = "0"+hextmp
                add_colors.append("#" + hextmp)
                tmp = tmp + step
            colors = colors + add_colors
    return colors

args = sys.argv
stdata = args[1]
mask = args[2]
outdir = args[3]
os.system("mkdir -p %s"%(outdir))
os.chdir(outdir)

adata = sc.read_h5ad(stdata)

## Define the cell type by choosing the largest deconvolution score
adata.obs.loc[:, 'annotated_cluster'] = adata.obs.loc[:, ['Tumor_Epithelial','Fibroblast','Myocyte','Endothelial','Macrophage','Tcell','Pericyte','Plasmocyte','DCS','Bcell','Normal_Epithelial','Mastcell']].idxmax(axis=1)
adata.obs['annotated_cluster'] = adata.obs['annotated_cluster'].astype('category')

## Define tumor region from processed image
sys.path.insert(
    0,
    'spatial_pipeline/lib'
)
obs_tags = 'annotated_cluster'
slice = 'P8'
tissue = 'ET'
cluster = 'Tumor_Epithelial'
tissue_mask = 'regist.tif'
view = False

indata = 'cellbin_annotated.h5ad'
outdir = 'image_extract'
outdata = 'cellbin_annotated_extract.h5ad'
mask = 'regist_mask_ft.tif'
cellbin_gem = 'data_adjust.txt'
os.system(f'mkdir -p {outdir}')
os.chdir(outdir)
figdir = './figures'
sc.settings.figdir = figdir
adata = sc.read_h5ad(indata)

os.system(f'gem_ext extr {tissue_mask} {cellbin_gem} -c red -e -o {outdir}')

extr_gem_fl = glob.glob(f'{outdir}/*.gem.gz')[0]
label = pd.read_csv(extr_gem_fl, sep='\t', comment='#')
label = label['label'].unique().astype('str')

adata.obs[obs_tags] = adata.obs[obs_tags].astype('category')
if tissue not in adata.obs[obs_tags].cat.categories:
    adata.obs[obs_tags] = adata.obs[obs_tags].cat.add_categories(tissue)
if view:
    print(adata.obs[obs_tags][adata.obs_names.isin(label)].value_counts())
    adata.obs[obs_tags][adata.obs_names.isin(label)].value_counts().to_csv(f'{outdir}/view.csv')
else:
    print(adata.obs[obs_tags][adata.obs_names.isin(label)].value_counts())
    if cluster == 'All':
        adata.obs[obs_tags][adata.obs_names.isin(label)] = tissue
    else:
        cluster_list = cluster.split(',')
        adata.obs[obs_tags][adata.obs_names.isin(label) & adata.obs[obs_tags].isin(cluster_list)] = tissue
        adata.obs[obs_tags][(~adata.obs_names.isin(label)) & adata.obs[obs_tags].isin(cluster_list)] = 'other'
    print(adata.obs[obs_tags][adata.obs_names.isin(label)].value_counts())
    #adata.obs[obs_tags] = adata.obs[obs_tags].cat.remove_unused_categories()
    adata.write(outdata)
    print("save tissue exract data done!")

adata.write(outdata)


## Plot results
res = pd.DataFrame(adata.obs, columns = ["x", "y", 'annotated_cluster'], index = adata.obs.index)
res.to_csv("bin1clu_celltype.txt",sep = '\t',index =False)
clusters = adata.obs['annotated_cluster'].cat.categories
cluster_number = clusters.shape[0]
colors = getDefaultColors(cluster_number, type = 2)
flout = open("color_celltype.list",'w')
for i in range(cluster_number):
    if clusters[i] == 'low_quality':
        flout.write(clusters[i] + '\t#ffffff\n')
    else:
        flout.write(clusters[i] + '\t' + colors[i] + '\n')
flout.close()
os.system('cell_bin_plot bin1clu_celltype.txt %s color_celltype.list celltype.tif'%(mask))
# Add legend to cell bin plot
Image.MAX_IMAGE_PIXELS = None
# Read color.list
categories = []
colors = []
with open('color_celltype.list', 'r') as f:
    for line in f:
        category, color = line.strip().split('\t')
        categories.append(category)
        colors.append(color)

im = Image.open('celltype.tif')
draw = ImageDraw.Draw(im)
font = ImageFont.truetype("arial.ttf", 500)
text_height = font.getsize('A')[1]
text_widths = [font.getsize(category)[0] for category in categories]
max_text_width = max(text_widths)
rect_width = max_text_width + 60
rect_height = (text_height + 10) * len(categories) + 10
rect_x = im.width - rect_width - 10
#rect_x = 900
rect_y = im.height - rect_height - 10
# draw.rectangle((rect_x, rect_y, rect_x+rect_width, rect_y+rect_height), fill='white')
for i in range(len(categories)):
    y = rect_y + i*(text_height+10) + 10
    draw.ellipse((rect_x+10-text_height, y, rect_x, y+text_height), fill=colors[i])
    draw.text((rect_x+40, y), categories[i], font=font, fill='white')
im.save('celltype_add_legend.tif')

## Split each cluster
os.system("mkdir -p cluster_split")
os.chdir("cluster_split")
flout = open("color.list", 'w')
flout.write('yes\t#ff0000\n')
flout.write('Low_quality\t#ffffff\n')
flout.close()
res['annotated_cluster'] = res['annotated_cluster'].astype('category').cat.add_categories('yes')
res['annotated_cluster'] = res['annotated_cluster'].astype('category').cat.add_categories('Low_quality')
for i in clusters:
    if i == 'Low_quality':
        continue
    tmp = res.copy()
    tmp.loc[tmp['annotated_cluster'] != i, ['annotated_cluster']] = 'Low_quality'
    tmp.loc[tmp['annotated_cluster'] == i, ['annotated_cluster']] = 'yes'
    tmp.to_csv("bin1clu_%s.txt" % (i), sep='\t', index=False)
    os.system(
        'cell_bin_plot bin1clu_%s.txt %s color.list cluster_plot_%s.tif'
        % (i, mask, i))
