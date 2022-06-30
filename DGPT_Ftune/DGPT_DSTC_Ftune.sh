# git clone -b patch-1 https://github.com/ghadiaravi13/DialogRPT.git
git clone https://github.com/bsantraigi/DialogRPT.git
git clone https://github.com/bsantraigi/2021-R3-Baselines
gdown --folder https://drive.google.com/drive/folders/1-5M_61VzH57TXWn1aKhjtk4yoBMnS4CY -O preprocessed_data/DSTC/
wget https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl -P restore
python DialogRPT/DGPT_Ftune/DGPT_Ftune_DSTC.py $1 $2 $3 $4 $5
