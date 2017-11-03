export PYTHONIOENCODING=UTF-8
export DATA_PATH=/home/guoyu/code/code_other_paper/conv_seq2seq/dataset/
export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.en
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.de
export DEV_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.en
export DEV_TARGETS=${DATA_PATH}/newstest2013.tok.bpe.32000.de
export DEV_TARGETS_REF=${DATA_PATH}/newstest2013.tok.de
export TEST_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.en
export TEST_TARGETS=${DATA_PATH}/newstest2013.tok.bpe.32000.de
export TRAIN_STEPS=1000000
export MODEL_DIR=/home/guoyu/code/code_other_paper/conv_seq2seq/model/nmt_conv_seq2seq
mkdir -p $MODEL_DIR