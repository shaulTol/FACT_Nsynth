


1.to get the code, clone the git:

git clone https://github.com/MediaBrain-SJTU/FACT.git
cd FACT

2. downloaded PACS from:

https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECz...

3.

in FACT/data i created a folder called images, within it another folde alled PACS and within this folder placed the kfold data folder.

so now the folder looks like this



4.
OPTIONAL - TRY THIS APPROACH WHEN FIRST APPROACH FAILS

downloaded the txt files for the split of train val and test from here. ONLY THE TXT FILES

https://drive.google.com/drive/folders/1i23DCs4TJ8LQsmBiMxsxo6qZsbhiX0gw


IMPORTANT: the downloaded file do not match the exact template used in our paper. for instance, the file names differ a bit:

in our paper:

art_painting_test.txt

in the downloded:

art_painting_test_kfold.txt

so need to remove the kfold part

In addition, the format of the text within the file differs as well:

our paper:

/home/user/data/images/PACS/kfold/art_painting/dog/pic_001.jpg 0

the downloaded files:

art_painting/dog/pic_225.jpg 1

so need to change to our format as well.

Chosen approach - change the downloaded file names to fit the paper's file names.

using command terminal change the text within the files to match the desired format.

lastly, replace all files in the datalists file with the new files we just edited.

used this to change all paths:

for file in /mnt/c/Users/stolk/OneDrive/github/FACT/AFourier-based-Framework-for-Domain-Generalization/data/datalists/*.txt; do
    sed -i 's|/Users/yaronot/FACT|/mnt/c/Users/stolk/OneDrive/github/FACT/AFourier-based-Framework-for-Domain-Generalization|g' "$file"
done


output for instance is:

/mnt/c/Users/stolk/OneDrive/github/FACT/AFourier-based-Framework-for-Domain-Generalization/data/images/PACS/kfold/art_painting/dog/pic_225.jpg 0

at the moment we didn't change the class labels.

the github mentions dog is labeled 0 while the downloaded format dog is labeled 1.

if issues occur later, try changing it as well and see if it helps.

we found later that there's indeed issue with the labeling so needed to decrease them by 1 to start from 0 with this code:

for file in *.txt; do
awk '{print $1, $2-1}' "$file" > tmp && mv tmp "$file"
done



5.

create a folder for checkpoints for training:

cd FACT

mkdir ckpt

6.

update shell_train.py path to

input_dir = '/mnt/c/Users/stolk/OneDrive/github/FACT/AFourier-based-Framework-for-Domain-Generalization/data/datalists'
output_dir = '/mnt/c/Users/stolk/OneDrive/github/FACT/AFourier-based-Framework-for-Domain-Generalization/train_logs'

notice to run it on unix enviorenment it required unix like path 

7.

create a virtual environment to make sure the proper versions are installed.

make sure to download the wheel of pytorch 1.1.0 because it wont run on brew for some reason. so we need to download it from here:
https://download.pytorch.org/whl/cpu/torch_stable.html
I chose :

torch-1.1.0-cp36-none-macosx_10_7_x86_64.whl

because I use mac, if you use windows choose the windows option.

after downlaoding it i put it in the FACT folder.

conda create --name fact_env python=3.6

conda activate fact_env

cd FACT

pip install torch-1.1.0-cp36-none-macosx_10_7_x86_64.whl

pip install torchvision==0.3.0

conda install opencv

pip install tensorflow - might need a suitable version (1.1.5 in my case)

pip install scipy

(if theres an issue do pip uninstall tensorflow before the rest - for some reason it worked afterwards)





9. now do the shell:

python shell_train.py -d=art_painting
