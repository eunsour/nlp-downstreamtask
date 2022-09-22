install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip list

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +

clean-test:
	-rm -r runs
	-rm -r outputs
	-rm -r cache_dir
	-rm -r wandb

formatter:
	black --line-length 119 generations tests 

types:
	pytype --keep-going generations --exclude generations/experimental