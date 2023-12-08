#!/bin/bash

: ${NODES:=1}

rm data
ln -s /home/s2/data data

salloc -N $NODES --exclusive --nodelist=a07 --partition=class1  --gres=gpu:4     \
  mpirun --bind-to none -mca btl ^openib -npernode 1              \
  numactl --physcpubind 0-63                                      \
  ./classifier $@
