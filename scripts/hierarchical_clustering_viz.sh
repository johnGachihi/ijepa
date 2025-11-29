IMG_IDX=23

python scripts/hierarchical_clustering_viz.py \
    --dataset sen2venus \
    --data-root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/sen2venus.hdf5 \
    --splits-file /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/splits_v1.json \
    --image-idx $IMG_IDX \
    --img-size 896 \
    --patch-size 14 \
    --n-clusters 3 4 5 10 \
    --sen2venus-image-type lr \
    --output-dir clustering_results_lrimg \
    --pretrained-checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_lr/jepa_sen2venus_lr-ep300.pth.tar


#python scripts/hierarchical_clustering_viz.py \
#    --dataset sen2venus \
#    --data-root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/sen2venus.hdf5 \
#    --splits-file /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/splits_v1.json \
#    --image-idx $IMG_IDX \
#    --img-size 896 \
#    --patch-size 14 \
#    --n-clusters 3 4 5 10 \
#    --sen2venus-image-type lr \
#    --output-dir clustering_results_hr_lrimg \
#    --pretrained-checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_hr/jepa_sen2venus-ep100.pth.tar

python scripts/hierarchical_clustering_viz.py \
    --dataset sen2venus \
    --data-root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/sen2venus.hdf5 \
    --splits-file /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus/splits_v1.json \
    --image-idx $IMG_IDX \
    --img-size 896 \
    --patch-size 14 \
    --n-clusters 3 4 5 10 \
    --sen2venus-image-type lr \
    --output-dir clustering_results_hr_lrwgram \
    --pretrained-checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_hr_gram/jepa_sen2venus_lr_hr_gram-ep300.pth.tar