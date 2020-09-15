# kaggle API and UI archives have different names

archived = 55175_105481_bundle_archive.zip

train: data/processed
	python model/model.py --path $^

data/processed: data/
	python model/data.py --raw $^ --out $@

data: $(archived)
	unzip 55175_105481_bundle_archive.zip -d data/

	@# Remove the duplicate file
	rm -rf data/yoochoose-data

	mv 55175_105481_bundle_archive.zip data/ 

$(archived):
	kaggle datasets download -d chadgostopp/recsys-challenge-2015
	mv recsys-challenge-2015.zip $@



.PHONY: train
