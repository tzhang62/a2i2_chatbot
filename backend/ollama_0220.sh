
# town person choices: bob, niki, ross, lindsay, michelle
python3 ollama_0220.py \
    -persona data_for_train/persona.json \
    -answer data_for_train/answers.json \
    -townperson 'ross' \
    --use-mps