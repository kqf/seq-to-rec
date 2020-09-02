
train: data/processed
	echo "To be implemented"

data/processed: data
	mkdir -p $@

data:
	@# Data can be downloaded from kaggle: 
	@# https://www.kaggle.com/chadgostopp/recsys-challenge-2015
	@# There's no rules, so kaggle api doesn't work

	unzip 55175_105481_bundle_archive.zip -d data/

	mv 55175_105481_bundle_archive.zip data/ 


.PHONY: train
