PY=python

prepare:
	$(PY) -m src.processing.prepare

base-eval:
	$(PY) -m src.eval.run_base_eval

train:
	$(PY) -m src.training.train

ft-eval:
	$(PY) -m src.eval.run_ft_eval

demo:
	$(PY) -m src.inference.demo_encode
