#!/bin/bash
wget https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label
wget https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label
mkdir datasets/
mv train_5500.label datasets/
mv TREC_10.label datasets/
# download papers
mkdir papers/
wget https://www.comp.nus.edu.sg/~leews/publications/p31189-zhang.pdf
mv p31189-zhang.pdf papaers/