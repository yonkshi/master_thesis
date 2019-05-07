up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python train.py

sync:
	rsync -arvu -f '- /*/' -e ssh . naturalreaders:masterthesis