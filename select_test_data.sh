#!/bin/sh
rm -fdr ./DATA/POSITIVE_TEST/* ./DATA/NEGATIVE_TEST/*
for fn in `find ./DATA/POSITIVES -name "*.png"|gshuf|head -n 1000`;do cp $fn ./DATA/POSITIVE_TEST;done
for fn in `find ./DATA/NEGATIVES -name "*.png"|gshuf|head -n 1000`;do cp $fn ./DATA/NEGATIVE_TEST;done
