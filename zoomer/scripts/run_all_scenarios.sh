#!/bin/bash
set -e
cd "$(dirname "$0")"
LOGDIR=/csse/research/contructive-learning/CAM-LDS/zoomer/results/logs
mkdir -p "$LOGDIR"

SCENARIOS="2 3 4 6 7"

echo "===== TECHNIQUE LEVEL ====="
for s in $SCENARIOS; do
    echo ">>> technique scenario $s: train"
    python3 Train_TTP_Recognition_Multilabel.py --scenario $s 2>&1 | tee "$LOGDIR/technique_train_scenario${s}.log"
    echo ">>> technique scenario $s: test"
    python3 Test_TTP_Recognition_Multilabel.py --scenario $s 2>&1 | tee "$LOGDIR/technique_test_scenario${s}.log"
done

echo "===== TACTIC LEVEL ====="
for s in $SCENARIOS; do
    echo ">>> tactic scenario $s: train"
    python3 Train_TTP_Recognition_Tactic.py --scenario $s 2>&1 | tee "$LOGDIR/tactic_train_scenario${s}.log"
    echo ">>> tactic scenario $s: test"
    python3 Test_TTP_Recognition_Tactic.py --scenario $s 2>&1 | tee "$LOGDIR/tactic_test_scenario${s}.log"
done

echo "===== ALL DONE ====="
