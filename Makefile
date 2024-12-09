.PHONY: install run_simple run_resnet run_preprocess clean

install:
	pip install -r requirements.txt

run_rf:
	python randomforest.py

run_simple:
	python ./models_we_tried/network_intrusion_detection.py

run_resnet:
	python ./models_we_tried/network_intrusion_detection_resnet.py

run_kmeans:
	python ./models_we_tried/network_intrusion_detection_kmeans.py

clean:
	rm -rf __pycache__

