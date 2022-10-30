git clone https://github.com/bsantraigi/DialogRPT.git
git clone https://github.com/bsantraigi/2021-R3-Baselines
gdown --folder https://drive.google.com/drive/folders/1bPoHKwpAIKx12msk2DnEDFrSLaUZuOj1 -O preprocessed_data/DD/

# Get preprocessed_data/DD_cc/ from somewhere!
cd DialogRPT/
wget https://github.com/yq-wen/overlapping-datasets/releases/download/v0.1/cleaned_dd.zip
unzip cleaned_dd.zip
mkdir -p data/ijcnlp_dailydialog_cc/train/
mkdir -p data/ijcnlp_dailydialog_cc/validation/
mkdir -p data/ijcnlp_dailydialog_cc/test/
mv cleaned_dd/dialogs/train_dialogs.txt data/ijcnlp_dailydialog_cc/train/dialogues_train.txt
mv cleaned_dd/dialogs/valid_dialogs.txt data/ijcnlp_dailydialog_cc/validation/dialogues_validation.txt
mv cleaned_dd/dialogs/test_dialogs.txt data/ijcnlp_dailydialog_cc/test/dialogues_test.txt
rm -rf cleaned_dd
rm cleaned_dd.zip
cd ../

python DialogRPT/DGPT_Ftune/prepare_ddcc.py ./DialogRPT/data/ijcnlp_dailydialog_cc ./DialogRPT/preprocessed_data/DD_cc

wget https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl -P restore
python DialogRPT/DGPT_Ftune/DGPT_Ftune_DD_cc.py $1 $2 $3 $4 $5