# Makefile

EXE=d2q9-bgk

CC=mpiicc
CFLAGS= -qopenmp -qopt-report=5 -qopt-report-phase=vec -Ofast -xHost -Wall -std=c99 
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x256.final_state.dat
REF_AV_VELS_FILE=check/128x256.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
