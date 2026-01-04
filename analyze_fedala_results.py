"""
Analyze FedALA variants benchmark results and generate summary tables.
"""

# Parse results from the benchmark output
results = {}

raw_data = """
Actor           | 5        | fedala            | 0.2986
Actor           | 5        | fedala_complete   | 0.3185
Actor           | 5        | fedala_momentum   | 0.2860
Actor           | 5        | fedala_prox       | 0.3185
Actor           | 5        | fedala_s          | 0.3059
Actor           | 5        | fedavg            | 0.2981
Actor           | 5        | fedprox           | 0.3112
Actor           | 10       | fedala            | 0.2791
Actor           | 10       | fedala_complete   | 0.3078
Actor           | 10       | fedala_momentum   | 0.2775
Actor           | 10       | fedala_prox       | 0.2958
Actor           | 10       | fedala_s          | 0.2858
Actor           | 10       | fedavg            | 0.2999
Actor           | 10       | fedprox           | 0.2994
Actor           | 20       | fedala            | 0.3090
Actor           | 20       | fedala_complete   | 0.2997
Actor           | 20       | fedala_momentum   | 0.2637
Actor           | 20       | fedala_prox       | 0.3003
Actor           | 20       | fedala_s          | 0.2992
Actor           | 20       | fedavg            | 0.3039
Actor           | 20       | fedprox           | 0.3075
Amazon-ratings  | 5        | fedala            | 0.4464
Amazon-ratings  | 5        | fedala_complete   | 0.4260
Amazon-ratings  | 5        | fedala_momentum   | 0.4434
Amazon-ratings  | 5        | fedala_prox       | 0.4317
Amazon-ratings  | 5        | fedala_s          | 0.4397
Amazon-ratings  | 5        | fedavg            | 0.4260
Amazon-ratings  | 5        | fedprox           | 0.4276
Amazon-ratings  | 10       | fedala            | 0.4526
Amazon-ratings  | 10       | fedala_complete   | 0.4373
Amazon-ratings  | 10       | fedala_momentum   | 0.4530
Amazon-ratings  | 10       | fedala_prox       | 0.4421
Amazon-ratings  | 10       | fedala_s          | 0.4565
Amazon-ratings  | 10       | fedavg            | 0.4314
Amazon-ratings  | 10       | fedprox           | 0.4325
Amazon-ratings  | 20       | fedala            | 0.4599
Amazon-ratings  | 20       | fedala_complete   | 0.4448
Amazon-ratings  | 20       | fedala_momentum   | 0.4570
Amazon-ratings  | 20       | fedala_prox       | 0.4516
Amazon-ratings  | 20       | fedala_s          | 0.4584
Amazon-ratings  | 20       | fedavg            | 0.4276
Amazon-ratings  | 20       | fedprox           | 0.4297
Chameleon       | 5        | fedala            | 0.5869
Chameleon       | 5        | fedala_complete   | 0.6084
Chameleon       | 5        | fedala_momentum   | 0.5826
Chameleon       | 5        | fedala_prox       | 0.5934
Chameleon       | 5        | fedala_s          | 0.5656
Chameleon       | 5        | fedavg            | 0.5717
Chameleon       | 5        | fedprox           | 0.5804
Chameleon       | 10       | fedala            | 0.5633
Chameleon       | 10       | fedala_complete   | 0.5674
Chameleon       | 10       | fedala_momentum   | 0.5801
Chameleon       | 10       | fedala_prox       | 0.5969
Chameleon       | 10       | fedala_s          | 0.5863
Chameleon       | 10       | fedavg            | 0.4939
Chameleon       | 10       | fedprox           | 0.5149
Chameleon       | 20       | fedala            | 0.5063
Chameleon       | 20       | fedala_complete   | 0.5158
Chameleon       | 20       | fedala_momentum   | 0.4856
Chameleon       | 20       | fedala_prox       | 0.5104
Chameleon       | 20       | fedala_s          | 0.5002
Chameleon       | 20       | fedavg            | 0.4720
Chameleon       | 20       | fedprox           | 0.4860
CiteSeer        | 5        | fedala            | 0.7010
CiteSeer        | 5        | fedala_complete   | 0.7233
CiteSeer        | 5        | fedala_momentum   | 0.7121
CiteSeer        | 5        | fedala_prox       | 0.7077
CiteSeer        | 5        | fedala_s          | 0.7099
CiteSeer        | 5        | fedavg            | 0.6987
CiteSeer        | 5        | fedprox           | 0.7002
CiteSeer        | 10       | fedala            | 0.7095
CiteSeer        | 10       | fedala_complete   | 0.7265
CiteSeer        | 10       | fedala_momentum   | 0.7162
CiteSeer        | 10       | fedala_prox       | 0.7199
CiteSeer        | 10       | fedala_s          | 0.7124
CiteSeer        | 10       | fedavg            | 0.7006
CiteSeer        | 10       | fedprox           | 0.7073
CiteSeer        | 20       | fedala            | 0.6690
CiteSeer        | 20       | fedala_complete   | 0.6856
CiteSeer        | 20       | fedala_momentum   | 0.6581
CiteSeer        | 20       | fedala_prox       | 0.6849
CiteSeer        | 20       | fedala_s          | 0.6689
CiteSeer        | 20       | fedavg            | 0.6603
CiteSeer        | 20       | fedprox           | 0.6581
Computers       | 5        | fedala            | 0.8800
Computers       | 5        | fedala_complete   | 0.8633
Computers       | 5        | fedala_momentum   | 0.8646
Computers       | 5        | fedala_prox       | 0.8729
Computers       | 5        | fedala_s          | 0.8766
Computers       | 5        | fedavg            | 0.8269
Computers       | 5        | fedprox           | 0.8521
Computers       | 10       | fedala            | 0.8712
Computers       | 10       | fedala_complete   | 0.8469
Computers       | 10       | fedala_momentum   | 0.8456
Computers       | 10       | fedala_prox       | 0.8678
Computers       | 10       | fedala_s          | 0.8734
Computers       | 10       | fedavg            | 0.7801
Computers       | 10       | fedprox           | 0.8071
Computers       | 20       | fedala            | 0.8602
Computers       | 20       | fedala_complete   | 0.8383
Computers       | 20       | fedala_momentum   | 0.8447
Computers       | 20       | fedala_prox       | 0.8604
Computers       | 20       | fedala_s          | 0.8667
Computers       | 20       | fedavg            | 0.7907
Computers       | 20       | fedprox           | 0.7774
Cora            | 5        | fedala            | 0.8260
Cora            | 5        | fedala_complete   | 0.8324
Cora            | 5        | fedala_momentum   | 0.8196
Cora            | 5        | fedala_prox       | 0.8305
Cora            | 5        | fedala_s          | 0.8251
Cora            | 5        | fedavg            | 0.8169
Cora            | 5        | fedprox           | 0.8169
Cora            | 10       | fedala            | 0.8152
Cora            | 10       | fedala_complete   | 0.8214
Cora            | 10       | fedala_momentum   | 0.8053
Cora            | 10       | fedala_prox       | 0.8197
Cora            | 10       | fedala_s          | 0.8116
Cora            | 10       | fedavg            | 0.8035
Cora            | 10       | fedprox           | 0.8098
Cora            | 20       | fedala            | 0.7704
Cora            | 20       | fedala_complete   | 0.7941
Cora            | 20       | fedala_momentum   | 0.7686
Cora            | 20       | fedala_prox       | 0.7950
Cora            | 20       | fedala_s          | 0.7747
Cora            | 20       | fedavg            | 0.7722
Cora            | 20       | fedprox           | 0.7739
Photo           | 5        | fedala            | 0.9123
Photo           | 5        | fedala_complete   | 0.9119
Photo           | 5        | fedala_momentum   | 0.9048
Photo           | 5        | fedala_prox       | 0.9155
Photo           | 5        | fedala_s          | 0.9194
Photo           | 5        | fedavg            | 0.8876
Photo           | 5        | fedprox           | 0.8856
Photo           | 10       | fedala            | 0.8975
Photo           | 10       | fedala_complete   | 0.8884
Photo           | 10       | fedala_momentum   | 0.8994
Photo           | 10       | fedala_prox       | 0.9007
Photo           | 10       | fedala_s          | 0.9046
Photo           | 10       | fedavg            | 0.8535
Photo           | 10       | fedprox           | 0.8593
Photo           | 20       | fedala            | 0.8944
Photo           | 20       | fedala_complete   | 0.8772
Photo           | 20       | fedala_momentum   | 0.8720
Photo           | 20       | fedala_prox       | 0.8919
Photo           | 20       | fedala_s          | 0.8896
Photo           | 20       | fedavg            | 0.8544
Photo           | 20       | fedprox           | 0.8707
PubMed          | 5        | fedala            | 0.8571
PubMed          | 5        | fedala_complete   | 0.8562
PubMed          | 5        | fedala_momentum   | 0.8556
PubMed          | 5        | fedala_prox       | 0.8582
PubMed          | 5        | fedala_s          | 0.8585
PubMed          | 5        | fedavg            | 0.8535
PubMed          | 5        | fedprox           | 0.8548
PubMed          | 10       | fedala            | 0.8532
PubMed          | 10       | fedala_complete   | 0.8307
PubMed          | 10       | fedala_momentum   | 0.8410
PubMed          | 10       | fedala_prox       | 0.8512
PubMed          | 10       | fedala_s          | 0.8541
PubMed          | 10       | fedavg            | 0.8184
PubMed          | 10       | fedprox           | 0.8239
PubMed          | 20       | fedala            | 0.8455
PubMed          | 20       | fedala_complete   | 0.8320
PubMed          | 20       | fedala_momentum   | 0.8393
PubMed          | 20       | fedala_prox       | 0.8488
PubMed          | 20       | fedala_s          | 0.8465
PubMed          | 20       | fedavg            | 0.8261
PubMed          | 20       | fedprox           | 0.8191
"""

# Parse the data
for line in raw_data.strip().split('\n'):
    if '|' in line:
        parts = [p.strip() for p in line.split('|')]
        dataset = parts[0]
        clients = int(parts[1])
        algo = parts[2]
        acc = float(parts[3])
        results[(dataset, clients, algo)] = acc

datasets = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "Chameleon", "Actor", "Amazon-ratings"]
client_counts = [5, 10, 20]
algorithms = ["fedavg", "fedprox", "fedala", "fedala_s", "fedala_prox", "fedala_momentum", "fedala_complete"]

print("="*100)
print("FEDALA VARIANTS BENCHMARK - SUMMARY ANALYSIS")
print("="*100)

# 1. Overall average by algorithm
print("\n" + "="*80)
print("1. OVERALL AVERAGE ACCURACY BY ALGORITHM")
print("="*80)
algo_avgs = {}
for algo in algorithms:
    accs = [results.get((ds, nc, algo), 0) for ds in datasets for nc in client_counts]
    avg = sum(accs) / len(accs)
    algo_avgs[algo] = avg
    print(f"{algo:<18}: {avg:.4f}")

print("\nRanking (best to worst):")
for i, (algo, avg) in enumerate(sorted(algo_avgs.items(), key=lambda x: -x[1]), 1):
    print(f"  {i}. {algo:<18}: {avg:.4f}")

# 2. Per-dataset comparison
print("\n" + "="*80)
print("2. ALGORITHM COMPARISON BY DATASET (Average across N=5,10,20)")
print("="*80)

for dataset in datasets:
    print(f"\n{dataset}:")
    print("-" * 50)
    ds_results = {}
    for algo in algorithms:
        accs = [results.get((dataset, nc, algo), 0) for nc in client_counts]
        avg = sum(accs) / len(accs)
        ds_results[algo] = avg

    # Sort by accuracy
    for algo, avg in sorted(ds_results.items(), key=lambda x: -x[1]):
        accs = [results.get((dataset, nc, algo), 0) for nc in client_counts]
        print(f"  {algo:<18}: {avg:.4f}  (N=5: {accs[0]:.4f}, N=10: {accs[1]:.4f}, N=20: {accs[2]:.4f})")

# 3. Win count analysis
print("\n" + "="*80)
print("3. WIN COUNT ANALYSIS (Best algorithm per dataset/client combination)")
print("="*80)

win_counts = {algo: 0 for algo in algorithms}
for dataset in datasets:
    for nc in client_counts:
        best_algo = max(algorithms, key=lambda a: results.get((dataset, nc, a), 0))
        win_counts[best_algo] += 1

print(f"\nTotal configurations: {len(datasets) * len(client_counts)}")
for algo, count in sorted(win_counts.items(), key=lambda x: -x[1]):
    print(f"  {algo:<18}: {count} wins ({100*count/(len(datasets)*len(client_counts)):.1f}%)")

# 4. Improvement over baselines
print("\n" + "="*80)
print("4. IMPROVEMENT OVER BASELINES")
print("="*80)

fedavg_avg = algo_avgs["fedavg"]
fedprox_avg = algo_avgs["fedprox"]
fedala_avg = algo_avgs["fedala"]

print(f"\nBaseline FedAvg: {fedavg_avg:.4f}")
print(f"Baseline FedProx: {fedprox_avg:.4f}")
print(f"Original FedALA: {fedala_avg:.4f}")
print()

for algo in ["fedala_s", "fedala_prox", "fedala_momentum", "fedala_complete"]:
    avg = algo_avgs[algo]
    imp_fedavg = (avg - fedavg_avg) / fedavg_avg * 100
    imp_fedala = (avg - fedala_avg) / fedala_avg * 100
    print(f"{algo:<18}: {avg:.4f} ({imp_fedavg:+.2f}% vs FedAvg, {imp_fedala:+.2f}% vs FedALA)")

# 5. Best performing variant by scenario
print("\n" + "="*80)
print("5. BEST FedALA VARIANT BY DATASET TYPE")
print("="*80)

homophilic = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers"]
heterophilic = ["Chameleon", "Actor", "Amazon-ratings"]

print("\nHomophilic graphs (Cora, CiteSeer, PubMed, Photo, Computers):")
homo_results = {algo: [] for algo in algorithms}
for ds in homophilic:
    for nc in client_counts:
        for algo in algorithms:
            homo_results[algo].append(results.get((ds, nc, algo), 0))
for algo in algorithms:
    avg = sum(homo_results[algo]) / len(homo_results[algo])
    print(f"  {algo:<18}: {avg:.4f}")

print("\nHeterophilic graphs (Chameleon, Actor, Amazon-ratings):")
hetero_results = {algo: [] for algo in algorithms}
for ds in heterophilic:
    for nc in client_counts:
        for algo in algorithms:
            hetero_results[algo].append(results.get((ds, nc, algo), 0))
for algo in algorithms:
    avg = sum(hetero_results[algo]) / len(hetero_results[algo])
    print(f"  {algo:<18}: {avg:.4f}")

# Save to file
with open("fedala_variants_analysis.txt", "w") as f:
    import sys
    from io import StringIO

    # Re-run analysis to capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    print("="*100)
    print("FEDALA VARIANTS BENCHMARK - SUMMARY ANALYSIS")
    print("="*100)

    print("\n" + "="*80)
    print("1. OVERALL AVERAGE ACCURACY BY ALGORITHM")
    print("="*80)
    for algo in algorithms:
        print(f"{algo:<18}: {algo_avgs[algo]:.4f}")

    print("\nRanking (best to worst):")
    for i, (algo, avg) in enumerate(sorted(algo_avgs.items(), key=lambda x: -x[1]), 1):
        print(f"  {i}. {algo:<18}: {avg:.4f}")

    print("\n" + "="*80)
    print("2. WIN COUNT ANALYSIS")
    print("="*80)
    for algo, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {algo:<18}: {count} wins")

    print("\n" + "="*80)
    print("3. IMPROVEMENT OVER BASELINES")
    print("="*80)
    for algo in ["fedala_s", "fedala_prox", "fedala_momentum", "fedala_complete"]:
        avg = algo_avgs[algo]
        imp_fedavg = (avg - fedavg_avg) / fedavg_avg * 100
        imp_fedala = (avg - fedala_avg) / fedala_avg * 100
        print(f"{algo:<18}: {avg:.4f} ({imp_fedavg:+.2f}% vs FedAvg, {imp_fedala:+.2f}% vs FedALA)")

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    f.write(output)

print("\n" + "="*80)
print("Analysis saved to fedala_variants_analysis.txt")
print("="*80)
