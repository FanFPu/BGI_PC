#### inferCNV copy number for each tumor samples####
library(rjags)
library(infercnv)
library(AnnoProbe)
library(tidyverse)
library(Seurat)
library(parallel)
library(Cairo)
library(limma)
options(bitmapType = "cairo")


args = commandArgs(T)
sample_i = args[1]

library(rjags)
library(infercnv)
library(AnnoProbe)
library(tidyverse)
library(Seurat)
library(parallel)
library(Cairo)
library(limma)
options(bitmapType = "cairo")


topdir = "./Single_inferCNV"
workdir = "./Single_inferCNV"
if(!dir.exists(workdir)){dir.create(workdir, recursive = TRUE)}
setwd(workdir)
load('seurat_merge_Epithelial.RData')
##normalize
seurat_merge <- NormalizeData(seurat_merge)


seurat_path = paste0(topdir, "/saveData/1.4.1_seurat_merge_infercnv.RData")
if (!file.exists(seurat_path)) {
    seurat_merge_infercnv <- subset(seurat_merge, subset = cellTypes_new %in% c('Epithelial','Tcell', 'Bcell','Macrophage', 'Endothelial'))
    seurat_lists = SplitObject(seurat_merge_infercnv, split.by = 'Patients') 
    save(seurat_lists, file = seurat_path)
}

load(seurat_path)
for (sample in names(seurat_lists)) {
    if(!dir.exists(paste0(workdir, '/', sample))){dir.create(paste0(workdir, '/', sample), recursive = TRUE)}
}

##run inferCNV
run_infercnv = function(sample) {
  cat(sample)
  options(bitmapType = "cairo")
  seurat_i = seurat_lists[[sample]]
  save(seurat_i, file=paste0(workdir, '/', sample, '/seurat_Epithe.RData'))
  print(table(seurat_i$cellTypes_new))
  # rm(seurat_lists)
  gc()
  mat_comb <- as.matrix(GetAssayData(seurat_i, assay="RNA",slot = "counts"))
  
  geneinfo <- annoGene(rownames(mat_comb), "SYMBOL", "human")
  geneinfo <- geneinfo[with(geneinfo,order(chr,start)),c(1,4:6)]
  geneinfo <- geneinfo[!duplicated(geneinfo[,1]),]
  rownames(geneinfo) <- geneinfo$SYMBOL
  
  ## for inferCNV the geneorder file just have 3 colums: chr, star, end, there is no name colum
  geneinfo$SYMBOL <- NULL
  geneinfo = geneinfo %>% mutate(chr_n=strsplit2(geneinfo$chr, split='chr')[,2]) %>% 
    mutate(chr_n = replace(chr_n, chr_n=='X', 23)) %>%
    mutate(chr_n = replace(chr_n, chr_n=='Y', 24)) %>%
    mutate(chr_n = replace(chr_n, chr_n=='M', 25)) %>%
    mutate(chr_n = as.numeric(chr_n)) %>%
    arrange(chr_n, start, end) %>% 
    select(-chr_n)

  
  seurat_i$cluster <- Idents(seurat_i)
  metada <- seurat_i@meta.data
  metadata <- metada[,c("cellTypes_new", "cellTypes_new")]
  colnames(metadata) <- c("cellType", "group")

  
  write.table(metadata, file=paste0(workdir,'/',sample,"/cellAnnotations.txt"), sep="\t", col.names = FALSE,quote = FALSE)
  write.table(geneinfo, file=paste0(workdir,'/',sample,"/gene_ordering_file.txt"), sep = "\t", col.names = FALSE, quote = FALSE)
  
  count_mat <- mat_comb[rownames(geneinfo),]
  rm(mat_comb)
  rm(seurat_i)
  rm(metada)
  gc()
  
  infercnv_obj <- CreateInfercnvObject(raw_counts_matrix = count_mat, 
                                       gene_order_file = paste0(workdir,'/',sample,"/gene_ordering_file.txt"), 
                                       delim = "\t", min_max_counts_per_cell = c(10, +Inf),
                                       ref_group_names = c("Tcell", "Bcell", "Macrophage", "Endothelial"),
                                       annotations_file = paste0(workdir,'/',sample,"/cellAnnotations.txt"))
  
  rm(count_mat)
  gc()
  save(infercnv_obj, file = paste0(workdir,'/',sample,"/infercnv_obj1.RData"))
  # perform infercnv operations to reveal cnv signal
  infercnv_obj = infercnv::run(infercnv_obj,
                               cutoff=0.1,  # use 1 for smart-seq, 0.1 for 10x-genomics
                               out_dir=sample,  # dir is auto-created for storing outputs
                               cluster_by_groups=F,   # cluster
                               scale_data=FALSE,
                               denoise=T, 
                               sd_amplifier=1.5,
                               HMM=T, 
                               #analysis_mode='subclusters',
                               output_format = "png",
                               num_threads=50
  )
  save(infercnv_obj, file = paste0(workdir,'/',sample,"/infercnv_obj2.RData"))
}

run_infercnv("HBCP21")
