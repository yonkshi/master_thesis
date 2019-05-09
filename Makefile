up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python3 train.py

sync:
	rsync -arvu -f '- /*/' -e ssh . naturalreaders:master_thesis

tb:
	tensorboard --logdir logs_v2 --host 0.0.0.0 --port 8081