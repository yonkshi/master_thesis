up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	rm -r checkpoints/
	rm log/*
	git pull
	python main.py