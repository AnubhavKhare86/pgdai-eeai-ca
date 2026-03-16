"""
week3
-----
Package containing all modules for CA 1: Design Decisions 1 and 2.

  config.py: shared constants
  data_selection.py      : CSV loading and label mapping
  noise_remover.py       : text cleaning
  text_representation.py : TF-IDF feature engineering
  data_preparation.py    : basic train/test split utility
  dealing_data_imbalance.py: rare-class-aware split utility
  data_object.py         : train/test container with rare-class handling
  base_model.py          : abstract model interface
  model_selection.py     : Random Forest factory helper
  random_forest_model.py : concrete BaseModel using RandomForest
  training.py            : fit utility wrapper
  testing_and_results.py : evaluation utility wrapper
  translation.py         : optional multilingual translation step
  chained_classifier.py  : DD1: chained multi-output classifier
  main.py  : DD1 entry point

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""
