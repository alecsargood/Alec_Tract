#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=1:00:00

#$ -S /bin/bash
#$ -j y
#$ -N test_TTL 
#$ -cwd 

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/ellie_TTL/logs
#$ -e /cluster/project2/CU-MONDAI/ellie_TTL/logs 

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

data_dir=/cluster/project2/CU-MONDAI/ellie_TTL/data/fibercup
results_dir=/cluster/project2/CU-MONDAI/ellie_TTL/


in_odf=${data_dir}/fodfs/fibercup_fodf.nii.gz
in_seed=${data_dir}/masks/fibercup_wm.nii.gz
in_mask=${data_dir}/masks/fibercup_wm.nii.gz
out_tractogram=/cluster/project2/CU-MONDAI/ellie_TTL/results/test.tck

ttl_track.py -f  ${in_odf} ${in_seed} ${in_mask} ${out_tractogram}
