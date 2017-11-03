# GCNN for video captioning

This is a tensorflow implementation of the [convolutional seq2seq model](https://arxiv.org/abs/1705.03122) released by Facebook. This model is orignially written via Torch/Lua in [Fairseq](https://github.com/facebookresearch/fairseq). Considering Lua is not that popular as python in the industry and research community, I re-implemente this model with Tensorflow/Python after carefully reading the paper details and Torch/Lua codebase.     

This implementation is based on the framework of [Google seq2seq project](https://github.com/google/seq2seq), which has a detailed [documentation](https://google.github.io/seq2seq/) on how to use this framework. In this conv seq2seq project, I implement the conv encoder, conv decoder, and attention mechanism, as well as other modules needed by the conv seq2seq model, which is not available in the original seq2seq project. 


## Requirement

- Python 2.7.0+
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.0+ (this version is strictly required)
- and their dependencies

Please follow [seq2seq project](https://google.github.io/seq2seq/) on how to install the Convolutional Sequence to Sequence Learning project. 
## How to use
For dataset, please follow [seq2seq nmt guides](https://google.github.io/seq2seq/nmt/) to prepare your dataset

The following is an example of how to run iwslt de-en translation task.
### Train
```
export PYTHONIOENCODING=UTF-8
export DATA_PATH="/mnt/data3/guoyuyu/datasets/MSVD/"
export FEAT_PATH=${DATA_PATH}
export TRAIN_FILE=${DATA_PATH}/MSVD_train_feat_ResNet152_pool5_2048_pair.tfrecords
export VALID_FILE=${DATA_PATH}/MSVD_valid_feat_ResNet152_pool5_2048.tfrecords
export TEST_FILE=${DATA_PATH}/MSVD_test_feat_ResNet152_pool5_2048.tfrecords
export VOCAB_TARGET=${DATA_PATH}/msvd_dict.bpe
export TRAIN_STEPS=1000000
export MODEL_DIR="./model/vcap_conv_seq2seq/"
mkdir -p $MODEL_DIR
export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

CUDA_VISIBLE_DEVICES=0  nohup  python -m bin.train \
  --config_paths="
      ./example_configs/vcap_conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: VideoCaptioningInputPipeline
    params:
      shuffle: True
      v_shape: [30,2048]
      files:
        - $TRAIN_FILE" \
  --input_pipeline_dev "
    class: VideoCaptioningInputPipeline_Test
    params:
      v_shape: [30,2048]
      files:
        - $TEST_FILE" \
  --batch_size 64 \
  --eval_batch_size 64 \
  --eval_every_n_steps 1000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR  > ./logfile/vcap_conseq2seq.log &

CUDA_VISIBLE_DEVICES=0   python -m bin.train \
  --config_paths="
      ./example_configs/vcap_conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: VideoCaptioningInputPipeline
    params:
      shuffle: True
      v_shape: [30,2048]
      files:
        - $TRAIN_FILE" \
  --input_pipeline_dev "
    class: VideoCaptioningInputPipeline
    params:
      v_shape: [30,2048]
      files:
        - $TEST_FILE" \
  --batch_size 64 \
  --eval_batch_size 64 \
  --eval_every_n_steps 2 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
  
  
#### decode with beam search
```
CUDA_VISIBLE_DEVICES=0 python -u -m bin.infer \
  --tasks "
    - class: VcapDecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: VideoCaptioningInputPipeline_Test   
    params:
      shuffle: False
      v_shape: [30,2048]
      files:
        - $TEST_FILE" > ${PRED_DIR}/predictions.txt
```

python -m bin.infer \
  --tasks "
    - class: VcapDecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: VideoCaptioningInputPipeline_Test
    params:
      shuffle: False
      v_shape: [30,2048]
      files:
        - $TEST_FILE" > ${PRED_DIR}/predictions.txt



python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

```

### Test

```
export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}
```

#### decode with greedy search
```
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

```

#### decode with beam search
```
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt
```

#### calculate BLEU score
```
./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt
```


For more detailed instructions, please refer to [seq2seq project](https://google.github.io/seq2seq/).


Issues and contributions are warmly welcome.  


