#!/bin/bash
# Quick script to check training progress

echo "=== Training Process Status ==="
ps aux | grep "train_model.py" | grep -v grep | head -1

echo ""
echo "=== Results Directory ==="
if [ -d "peptide_full_results" ]; then
    echo "Directory exists"
    ls -lh peptide_full_results/ 2>/dev/null | head -10
    echo ""
    echo "File sizes:"
    du -sh peptide_full_results/* 2>/dev/null | head -5
else
    echo "Results directory not created yet (still initializing)"
fi

echo ""
echo "=== Recent Log Output ==="
tail -20 peptide_training.log 2>/dev/null || echo "Log file empty or not found"

