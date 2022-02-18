
********** Build Your Own Transformer **********

***** Download Cleaned Data *****

Please download the preprocessed data from: https://drive.google.com/file/d/1abyxItvZ1g1J_qKi8tcsVXeVpr4EJrfl/view?usp=sharing

If data is downloaded, no need for the next step (Moses Tokenizer)

***** Run Moses Tokenizer *****

Needed only when using unpreprocessed raw data from: http://www.statmt.org/wmt13/training-parallel-nc-v8.tgz

Please download Moses Tokenizer from: https://drive.google.com/file/d/1zQ9dbw0Y5veEdglPN4VgPUEBJaJCHMVR/view?usp=sharing

Copy downloaded Moses Tokeizer folder in the data folder path

Run the following scripts on the data downloaded:

-> perl dir_name/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en < data/train.en > data/train.norm.en

-> perl dir_name/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en < data/train.norm.en > data/train.norm.tok.en

-> perl path/to/mosesdecoder/scripts/training/clean-corpus-n.perl data/train.norm.tok en de data/train.norm.tok.clean 1 80

***** Prepare Code Environment *****

Please create conda environment from "environment.yml" using the following command:

conda env create -f environment.yml

***** Run Source Code *****

The code entry to run is "transformer.py"

Please specify the data path folder and name in the first two lines

More information on the code structure is shown in the attached presentation

********** End **********