#! /bin/bash
## The first line has to say this, as a bash script
## Job Name
#SBATCH--job-name=mz_project
## Partition and Allocation
#SBATCH -p stf
#SBATCH -A stf
## Resources
#SBATCH --nodes=1
#SBATCH--time=0:60:00
#SBATCH --ntasks=1
#SBATCH --mem=10g
## Specify the working directory for this job
#SBATCH --chdir=.
## Import any modules here
## module load contrib/anaconda-2019.03
## Scripts to be executed here 
export PATH="/gscratch/stf/mz/anaconda3/bin:$PATH"
conda activate paddle_env
cd project
python PaddleOCR/tools/train.py -c configs/vi_layoutxlm.yml
## Clean up
exit 0