# kaggle API and UI archives have different names

archived = 55175_105481_bundle_archive.zip

train: data/processed
	python model/model.py --path $^

data/processed: data/
	python model/data.py --raw $^ --out $@

data/: 
	kaggle datasets download -d chadgostopp/recsys-challenge-2015

	mv recsys-challenge-2015.zip $(archived)
	mkdir -p data/
	unzip $(archived) -d data/

	@# Remove the duplicate file
	rm -rf data/yoochoose-data

	mv $(archived) data/ 



clean:
	rm -rf $(archived)

.PHONY: train
