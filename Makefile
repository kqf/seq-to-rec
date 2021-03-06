# kaggle API and UI archives have different names

archived = 55175_105481_bundle_archive.zip

train: data/processed
	python model/flat/rnn.py --path $^

data/processed: data/
	python model/data.py --raw $^ --out $@
	@echo "Training dataset:"
	@head data/processed/train.txt

data/: 
	kaggle datasets download -d chadgostopp/recsys-challenge-2015

	mv recsys-challenge-2015.zip $(archived)
	unzip $(archived) -d data/

	@# Remove the duplicate file
	rm -rf data/yoochoose-data

	mv $(archived) data/ 



clean:
	rm -rf data/processed/

.PHONY: train
