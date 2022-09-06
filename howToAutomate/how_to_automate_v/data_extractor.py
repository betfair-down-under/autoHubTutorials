# Extracts all the bzip2 files (.bz2) 
# contained within the specified output_folder and any sub-folders
# and writes them to a file with their market_id as their file name
# This will take around 10 mins to run for one month of Pro Greyhound data
import glob
import bz2
import shutil

# # Folder containing bz2 files or any subfolders with bz2 files
input_folder = '2022_04_AprGreyhoundsPro'
# input_folder = 'sample_monthly_data'
# # Folder to write our extracted bz2 files to, this folder needs to already be created
output_folder = 'output_2022_04_AprGreyhoundsPro'
# output_folder = 'sample_monthly_data_output'

# Returns a list of paths to bz2 files within the input folder and any sub folders
files = glob.iglob(f'{input_folder}/**/**/**/**/**/*.bz2', recursive = False)

# Extracts each bz2 file and write it to the output folder
for path in files:
    market_id = path[-15:-4]
    print(path, market_id)
    with bz2.BZ2File(path) as fr, open(f'{output_folder}/{market_id}',"wb") as fw:
        shutil.copyfileobj(fr,fw)