# Examples taken from https://github.com/Unbabel/COMET
echo -e "Dem Feuer konnte Einhalt geboten werden\nSchulen und Kindergärten wurden eröffnet." >> src.de
echo -e "The fire could be stopped\nSchools and kindergartens were open" >> hyp1.en
echo -e "They were able to control the fire.\nSchools and kindergartens opened" >> ref.en

# Run a reference-based version
comet-score -s src.de -t hyp1.en -r ref.en --gpus 0

# Run a reference-free version
comet-score -s src.de -t hyp1.en --gpus 0 --model wmt20-comet-qe-da

rm src.de hyp1.en ref.en