#!/bin/bash
# Check benchmark progress

echo "=== Benchmark Progress ==="
echo ""

# Check if running
if ps aux | grep benchmark_general.py | grep -v grep > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: FINISHED (or not started)"
fi

echo ""
echo "=== Current Progress ==="
grep -E "^(>>>|Benchmark Dataset|Result:)" benchmark_output.log 2>/dev/null | tail -10

echo ""
echo "=== Completed Experiments ==="
completed=$(grep -c "Result:" benchmark_output.log 2>/dev/null || echo 0)
echo "Completed: $completed / 168 experiments"

echo ""
echo "=== Latest Activity ==="
tail -5 benchmark_output.log 2>/dev/null
