.PHONY: install run_simple run_resnet run_preprocess clean

install:
	pip install -r requirements.txt

run_simple:
	python network_intrusion_detection.py

run_resnet:
	python network_intrusion_detection_resnet.py

run_kmeans:
	python network_intrusion_detection_kmeans.py

clean:
	rm -rf __pycache__

